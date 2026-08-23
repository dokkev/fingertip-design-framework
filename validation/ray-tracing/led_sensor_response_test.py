"""Compare bounded LED/receiver response before and after indentation."""

from __future__ import annotations

from importlib.resources import as_file, files
from time import perf_counter

import numpy as np
import warp as wp

from lumo.fingertip import (
    DRAGON_SKIN_10_NV_OPTICS_HIGH,
    DRAGON_SKIN_10_NV_OPTICS_LOW,
    DRAGON_SKIN_10_NV_OPTICS_NOMINAL,
    SOLARIS_OPTICS_HIGH,
    SOLARIS_OPTICS_LOW,
    SOLARIS_OPTICS_NOMINAL,
    Fingertip,
    FingertipParameters,
    SiliconeOptics,
)
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
_BOUNCE_CAP = 24
_RNG_SEED = 20260823

_OPTICAL_CASES: tuple[tuple[str, SiliconeOptics], ...] = (
    ("low", SOLARIS_OPTICS_LOW),
    ("nominal", SOLARIS_OPTICS_NOMINAL),
    ("high", SOLARIS_OPTICS_HIGH),
    ("low", DRAGON_SKIN_10_NV_OPTICS_LOW),
    ("nominal", DRAGON_SKIN_10_NV_OPTICS_NOMINAL),
    ("high", DRAGON_SKIN_10_NV_OPTICS_HIGH),
)

# Hardware optical defaults come from FingertipParameters.led. Placement remains
# an uncalibrated validation-local point-source fixed in the carrier frame.
_SOURCE_POSITION_W_M = np.array((-5.0e-3, 0.0, -20.0e-3))
_SOURCE_NORMAL_W = np.array((0.0, 0.0, 1.0))
_RECEIVER_CENTER_W_M = np.array((-10.0e-3, 0.0, -20.0e-3))
_RECEIVER_SIZE_XY_M = np.array((6.0e-3, 4.0e-3))

_SIM_FREQUENCY_HZ = 1.0e3
_SPHERE_RADIUS_M = 5.0e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_APPROACH_SPEED_M_S = 2.5e-2
_TARGET_FORCE_N = 20.0
_FORCE_TOLERANCE_N = 5.0
_SETTLE_DURATION_S = 5.0e-3
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


def _trace_response(
    scene: OptixScene,
    emission: np.ndarray,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
    *,
    optics: SiliconeOptics,
) -> dict[str, float | int]:
    result = trace_bounded_paths(
        scene,
        emission["origin_W_m"],
        emission["direction_W"],
        emission["power"],
        inside_silicone=False,
        n_air=_N_AIR,
        n_silicone=optics.refractive_index,
        extinction_coefficient_m_inv=optics.extinction_coefficient_m_inv,
        carrier_albedo=_CARRIER_ALBEDO,
        max_bounces=_BOUNCE_CAP,
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
        silicone_instance_id=SILICONE_INSTANCE_ID,
        carrier_instance_id=CARRIER_INSTANCE_ID,
        mask=ALL_MASK,
    )
    received_power, received_count, escaped_not_received = (
        _ideal_receiver_response(result.escaped_rays)
    )
    response: dict[str, float | int] = {
        "emitted_power": result.emitted_power,
        "escaped_power": result.escaped_power,
        "absorbed_power": result.absorbed_power,
        "bulk_loss_power": result.bulk_loss_power,
        "unresolved_internal_miss_power": result.unresolved_internal_miss_power,
        "remaining_power": result.remaining_power,
        "accounted_power": result.accounted_power,
        "closure_error": result.closure_error,
        "escaped_ray_count": result.escaped_ray_count,
        "remaining_ray_count": result.remaining_ray_count,
        "received_power": received_power,
        "received_ray_count": received_count,
        "escaped_not_received_power": escaped_not_received,
    }
    if not np.isclose(
        received_power + escaped_not_received,
        result.escaped_power,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("ideal receiver did not classify escaped power")
    return response


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
        parameters=fingertip.parameters.led,
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
        led.parameters.normalized_power,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("LED ray powers do not sum to source power")

    ray_count = len(emission)
    rng = np.random.default_rng(_RNG_SEED)
    dielectric_branch_u = rng.random((_BOUNCE_CAP, ray_count))
    carrier_u1 = rng.random((_BOUNCE_CAP, ray_count))
    carrier_u2 = rng.random((_BOUNCE_CAP, ray_count))

    undeformed: dict[tuple[str, str], dict[str, float | int]] = {}
    for assumption, optics in _OPTICAL_CASES:
        undeformed[(optics.name, assumption)] = _trace_response(
            scene,
            emission,
            dielectric_branch_u,
            carrier_u1,
            carrier_u2,
            optics=optics,
        )
    deformed: dict[tuple[str, str], dict[str, float | int]] | None = None

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
        deformed = {}
        for assumption, optics in _OPTICAL_CASES:
            deformed[(optics.name, assumption)] = _trace_response(
                scene,
                emission,
                dielectric_branch_u,
                carrier_u1,
                carrier_u2,
                optics=optics,
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
            motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
            approach_speed_m_s=_APPROACH_SPEED_M_S,
            target_force_n=_TARGET_FORCE_N,
            max_sim_time_s=_MAX_SIM_TIME_S,
        )
        DesignStudy(
            fingertip,
            (trial,),
            sim_frequency=_SIM_FREQUENCY_HZ,
            force_tolerance_n=_FORCE_TOLERANCE_N,
            settle_duration_s=_SETTLE_DURATION_S,
            max_search_iterations=_MAX_SEARCH_ITERATIONS,
        ).run(inspect_trial=inspect_deformed_trial)

    if deformed is None or trial.reaction_force_n is None:
        raise AssertionError("deformed optical response was not evaluated")

    print("Green Sequin LED silicone-optics sensitivity validation")
    print(
        "hardware: Adafruit Product "
        f"{led.parameters.ADAFRUIT_PRODUCT_ID} | "
        f"{led.parameters.LED_PART_NUMBER} | {led.parameters.PACKAGE}"
    )
    print(
        "spectral metadata: dominant="
        f"{led.parameters.dominant_wavelength_nm:g} nm | "
        f"peak={led.parameters.peak_wavelength_nm:g} nm | "
        f"half-width={led.parameters.spectral_half_width_nm:g} nm | "
        "viewing half-angle="
        f"{led.parameters.viewing_half_angle_deg:g} deg"
    )
    print(
        "source model: ideal point Lambertian at (-5, 0, -20) mm | "
        f"rays={ray_count} | "
        f"normalized P={led.parameters.normalized_power:g} | uncalibrated"
    )
    print(
        "receiver: ideal planar 6 x 4 mm aperture centered at "
        "(-10, 0, -20) mm | validation-local"
    )
    print(
        f"transport: wavelength=525 nm | n_air={_N_AIR:g} | "
        f"carrier_albedo={_CARRIER_ALBEDO:g} placeholder | "
        f"bounce_cap={_BOUNCE_CAP} | common samples seed={_RNG_SEED}"
    )
    print(
        "optics: Solaris n is manufacturer data; Dragon Skin n and all "
        "attenuation values are literature sensitivity priors, not product "
        "calibration"
    )
    print()

    relative_changes: dict[tuple[str, str], float] = {}
    for assumption, optics in _OPTICAL_CASES:
        key = (optics.name, assumption)
        undeformed_response = undeformed[key]
        deformed_response = deformed[key]
        received_undeformed = float(undeformed_response["received_power"])
        received_deformed = float(deformed_response["received_power"])
        relative_change = (
            (received_deformed - received_undeformed) / received_undeformed
            if received_undeformed > 0.0
            else float("nan")
        )
        relative_changes[key] = relative_change

        for response in (undeformed_response, deformed_response):
            if not np.all(
                np.isfinite(
                    [
                        float(response["received_power"]),
                        float(response["absorbed_power"]),
                        float(response["bulk_loss_power"]),
                        float(response["remaining_power"]),
                        float(response["unresolved_internal_miss_power"]),
                        float(response["accounted_power"]),
                    ]
                )
            ):
                raise AssertionError("bounded response contains invalid power")
            if not np.isclose(
                float(response["accounted_power"]),
                led.parameters.normalized_power,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise AssertionError("bounded response energy closure failed")
            if float(response["bulk_loss_power"]) <= 0.0:
                raise AssertionError("positive extinction produced no bulk loss")
        if not np.isfinite(relative_change):
            raise AssertionError("material case produced no receiver response")

        print(
            f"{optics.name:18s} {assumption:7s} | "
            f"n={optics.refractive_index:.4f} | "
            f"mu={optics.extinction_coefficient_m_inv:9.6f} 1/m | "
            f"P_recv U/D={received_undeformed:.9f}/"
            f"{received_deformed:.9f} | relative={relative_change:+.3%}"
        )
        print(
            "  bulk_loss U/D="
            f"{float(undeformed_response['bulk_loss_power']):.9f}/"
            f"{float(deformed_response['bulk_loss_power']):.9f} | "
            "carrier_absorbed U/D="
            f"{float(undeformed_response['absorbed_power']):.9f}/"
            f"{float(deformed_response['absorbed_power']):.9f} | "
            "remaining U/D="
            f"{float(undeformed_response['remaining_power']):.9f}/"
            f"{float(deformed_response['remaining_power']):.9f} | "
            "closure U/D="
            f"{float(undeformed_response['accounted_power']):.9f}/"
            f"{float(deformed_response['accounted_power']):.9f}"
        )

    response_signs = {np.sign(value) for value in relative_changes.values()}
    print()
    print(
        "deformation response range: "
        f"{min(relative_changes.values()):+.3%} to "
        f"{max(relative_changes.values()):+.3%} | "
        f"common sign={'yes' if len(response_signs) == 1 else 'no'}"
    )
    print()
    print(
        f"mechanics: F={trial.reaction_force_n:.4f} N | "
        f"travel={1.0e3 * float(trial.travel_m):.4f} mm | "
        f"t={float(trial.simulation_time_s):.3f} s"
    )
    print(f"wall runtime: {perf_counter() - wall_start:.3f} s")
    print()
    print("Undeformed/deformed silicone-optics sensitivity: PASS")


if __name__ == "__main__":
    main()
