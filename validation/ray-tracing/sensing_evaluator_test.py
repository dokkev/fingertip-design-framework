"""Validate side-view sensing objectives on deterministic contact states."""

from __future__ import annotations

from importlib.resources import as_file, files
from time import perf_counter

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.fingertip.geometric_param import semiellipse_depth_at_x_mm
from lumo.mesh import FingertipMesh, make_fingertip_mesh
from lumo.newton import Indenter
from lumo.optimization import sensing_descriptors, sensing_objectives
from lumo.ray_tracing import (
    LED,
    OptixScene,
    side_view_observation,
    trace_bounded_paths,
)
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK

_SOURCE_Y_FRACTIONS = (-0.55, 0.0, 0.55)
_SOURCE_Z_FRACTIONS = (0.15, 0.65)
_SOURCE_CLEARANCE_M = 1.0e-3
_SAMPLE_SIDE_COUNT = 32
_BOUNCE_CAP = 24
_RNG_SEED = 20260823
_CARRIER_ALBEDO = 0.7

_CONTACT_X_MM = (-7.5, 0.0, 7.5)
_SPHERE_RADIUS_M = 5.0e-3
_INITIAL_CLEARANCE_M = 1.0e-3
_TRANSLATION_STEP_M = 2.5e-5
_TARGET_FORCE_N = 20.0
_FORCE_TOLERANCE_N = 5.0
_FORCE_DURATION_S = 5.0e-3
_MAX_SIM_TIME_S = 30.0
_MAX_SEARCH_ITERATIONS = 256
_SIM_FREQUENCY_HZ = 1.0e3


def _make_leds(
    fingertip: Fingertip,
    fingertip_mesh: FingertipMesh,
) -> tuple[LED, ...]:
    """Return 12 validation-local sources outside the straight sidewalls."""
    vertices = np.asarray(fingertip_mesh.silicone.vertices, dtype=np.float64)
    half_extrusion_m = 0.5 * float(vertices[:, 1].max() - vertices[:, 1].min())
    side_x_m = 1.0e-3 * fingertip.silicone.half_width_mm + _SOURCE_CLEARANCE_M
    bottom_z_mm = fingertip.silicone.ellipse_center_z_mm
    side_height_mm = fingertip.silicone.bond_top_z_mm - bottom_z_mm
    leds = []
    for y_fraction in _SOURCE_Y_FRACTIONS:
        for z_fraction in _SOURCE_Z_FRACTIONS:
            z_m = 1.0e-3 * (bottom_z_mm + z_fraction * side_height_mm)
            for x_m, normal_x in ((-side_x_m, 1.0), (side_x_m, -1.0)):
                leds.append(
                    LED(
                        position_W_m=np.array(
                            (x_m, y_fraction * half_extrusion_m, z_m)
                        ),
                        normal_W=np.array((normal_x, 0.0, 0.0)),
                        parameters=fingertip.parameters.led,
                    )
                )
    return tuple(leds)


def _assert_external_sources(
    scene: OptixScene,
    leds: tuple[LED, ...],
    *,
    state_label: str,
) -> None:
    origins = np.stack([led.position_W_m for led in leds])
    inward = np.stack([led.normal_W for led in leds])
    inward_hits = scene.trace_closest(
        origins,
        inward,
        mask=_SILICONE_MASK,
    )["hit"]
    outward_hits = scene.trace_closest(
        origins,
        -inward,
        mask=_SILICONE_MASK,
    )["hit"]
    if not np.all(inward_hits) or np.any(outward_hits):
        raise AssertionError(f"{state_label} invalidated external source poses")


def _trace_observation(
    scene: OptixScene,
    fingertip: Fingertip,
    emission: np.ndarray,
    *,
    rays_per_led: int,
    dielectric_branch_u: np.ndarray,
    carrier_u1: np.ndarray,
    carrier_u2: np.ndarray,
) -> np.ndarray:
    optics = fingertip.parameters.optical
    escaped, statistics = trace_bounded_paths(
        scene,
        emission["origin_W_m"],
        emission["direction_W"],
        emission["power"],
        inside_silicone=False,
        n_air=1.0,
        n_silicone=optics.refractive_index,
        extinction_coefficient_m_inv=optics.extinction_coefficient_m_inv,
        carrier_albedo=_CARRIER_ALBEDO,
        max_bounces=_BOUNCE_CAP,
        dielectric_branch_u=dielectric_branch_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        mask=_ALL_MASK,
    )
    if not np.isclose(
        float(statistics["accounted_power"]),
        float(statistics["emitted_power"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("bounded transport failed energy closure")

    # The validation-local sources are external. Exclude rays that never
    # interacted with the fingertip from its side-view optical response.
    escaped = escaped[escaped["bounce"] > 0]
    led_index = escaped["ray_id"] // rays_per_led
    escaped_by_led = tuple(
        escaped[led_index == index]
        for index in range(len(emission) // rays_per_led)
    )
    return side_view_observation(
        escaped_by_led,
        fingertip=fingertip,
    )


def _make_trial(
    fingertip: Fingertip,
    sphere_urdf_path,
    *,
    contact_x_mm: float,
) -> DesignTrial:
    surface_z_mm = (
        fingertip.silicone.ellipse_center_z_mm
        - semiellipse_depth_at_x_mm(
            half_width_mm=fingertip.silicone.ellipse_radius_x_mm,
            height_mm=fingertip.silicone.ellipse_radius_z_mm,
            x_mm=contact_x_mm,
        )
    )
    initial_center_z_m = (
        1.0e-3 * surface_z_mm
        - _INITIAL_CLEARANCE_M
        - _SPHERE_RADIUS_M
    )
    return DesignTrial(
        name=f"contact_x={contact_x_mm:+.1f}mm",
        urdf_path=sphere_urdf_path,
        initial_tf=wp.transform(
            wp.vec3(1.0e-3 * contact_x_mm, 0.0, initial_center_z_m),
            wp.quat_identity(),
        ),
        translation_step_W_m=wp.vec3(0.0, 0.0, _TRANSLATION_STEP_M),
        target_force_n=_TARGET_FORCE_N,
        max_sim_time_s=_MAX_SIM_TIME_S,
    )


def _minimum_pair(descriptors: np.ndarray) -> tuple[float, tuple[int, int]]:
    best_distance = float("inf")
    best_pair = (-1, -1)
    for first in range(len(descriptors) - 1):
        for second in range(first + 1, len(descriptors)):
            distance = float(
                np.linalg.norm(descriptors[first] - descriptors[second])
            )
            if distance < best_distance:
                best_distance = distance
                best_pair = (first, second)
    return best_distance, best_pair


def main() -> None:
    wall_start = perf_counter()
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_mesh(fingertip)
    scene = OptixScene(
        fingertip_mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )

    leds = _make_leds(fingertip, fingertip_mesh)
    _assert_external_sources(scene, leds, state_label="no_contact")
    sample_i, sample_j = np.meshgrid(
        np.arange(_SAMPLE_SIDE_COUNT),
        np.arange(_SAMPLE_SIDE_COUNT),
        indexing="ij",
    )
    emission_u1 = (sample_i.ravel() + 0.5) / _SAMPLE_SIDE_COUNT
    emission_u2 = (sample_j.ravel() + 0.5) / _SAMPLE_SIDE_COUNT
    emission = np.concatenate(
        [led.emit(emission_u1, emission_u2) for led in leds]
    )
    rays_per_led = len(emission_u1)
    rng = np.random.default_rng(_RNG_SEED)
    dielectric_branch_u = rng.random((_BOUNCE_CAP, len(emission)))
    carrier_u1 = rng.random((_BOUNCE_CAP, len(emission)))
    carrier_u2 = rng.random((_BOUNCE_CAP, len(emission)))

    labels = ["no_contact"]
    responses = [
        _trace_observation(
            scene,
            fingertip,
            emission,
            rays_per_led=rays_per_led,
            dielectric_branch_u=dielectric_branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
        )
    ]

    def inspect_contact(
        trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        if simulation.soft_contact_count(indenter.body_index) == 0:
            raise AssertionError(f"{trial.name} has no indenter contact")
        vertices = simulation.silicone_vertices()
        if not np.all(np.isfinite(vertices)):
            raise AssertionError(f"{trial.name} has non-finite silicone")
        reference_vertices = np.asarray(
            fingertip_mesh.silicone.vertices,
            dtype=np.float32,
        )
        trial_reference_vertices = np.asarray(
            simulation.fingertip_mesh.silicone.vertices,
            dtype=np.float32,
        )
        if (
            trial_reference_vertices.shape != reference_vertices.shape
            or not np.allclose(
                trial_reference_vertices,
                reference_vertices,
                rtol=0.0,
                atol=1.0e-7,
            )
        ):
            raise AssertionError(f"{trial.name} changed silicone vertex ordering")
        if not np.array_equal(
            simulation.fingertip_mesh.silicone.surface_tri_indices,
            fingertip_mesh.silicone.surface_tri_indices,
        ):
            raise AssertionError(f"{trial.name} changed silicone topology")
        scene.update_silicone(vertices)
        _assert_external_sources(scene, leds, state_label=trial.name)
        if trial.final_tf is None:
            raise AssertionError(f"{trial.name} has no final indenter pose")
        sphere_top_z_m = float(np.asarray(trial.final_tf)[2]) + _SPHERE_RADIUS_M
        if min(led.position_W_m[2] for led in leds) <= sphere_top_z_m:
            raise AssertionError(f"{trial.name} sphere reached the source plane")
        labels.append(trial.name)
        responses.append(
            _trace_observation(
                scene,
                fingertip,
                emission,
                rays_per_led=rays_per_led,
                dielectric_branch_u=dielectric_branch_u,
                carrier_u1=carrier_u1,
                carrier_u2=carrier_u2,
            )
        )

    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_10mm.urdf",
    )
    with as_file(sphere_resource) as sphere_urdf_path:
        trials = tuple(
            _make_trial(
                fingertip,
                sphere_urdf_path,
                contact_x_mm=contact_x_mm,
            )
            for contact_x_mm in _CONTACT_X_MM
        )
        DesignStudy(
            fingertip,
            trials,
            sim_frequency=_SIM_FREQUENCY_HZ,
            force_tolerance_n=_FORCE_TOLERANCE_N,
            force_duration_s=_FORCE_DURATION_S,
            max_search_iterations=_MAX_SEARCH_ITERATIONS,
        ).run(inspect_trial=inspect_contact)

    response_array = np.stack(responses)
    if response_array.shape != (4, 12, 4):
        raise AssertionError("side-view response does not have shape (4, 12, 4)")
    intensity, spatial = sensing_descriptors(response_array)
    objective_intensity, objective_spatial = sensing_objectives(response_array)
    measured_intensity, intensity_pair = _minimum_pair(intensity)
    measured_spatial, spatial_pair = _minimum_pair(spatial)
    if not np.isclose(objective_intensity, measured_intensity) or not np.isclose(
        objective_spatial,
        measured_spatial,
    ):
        raise AssertionError("objective and reported worst-case pair disagree")
    if not np.allclose(intensity[0], 0.0, rtol=0.0, atol=1.0e-15):
        raise AssertionError("no-contact intensity descriptor is not zero")
    if not np.allclose(spatial.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("spatial descriptors are not normalized")

    np.set_printoptions(precision=7, suppress=True, linewidth=160)
    print("LUMO side-view sensing evaluator")
    print("side convention: +Y; quadrants Q1,Q2,Q3,Q4 in canonical X-Z")
    print(
        "sources: validation-local 2 sides x 3 Y x 2 Z grid | "
        "1.0 mm outside straight sidewalls | interacted paths only | "
        f"rays={rays_per_led} per LED | bounce_cap={_BOUNCE_CAP}"
    )
    print(
        f"optics: {fingertip.parameters.optical.name} nominal | "
        f"n={fingertip.parameters.optical.refractive_index:g} | "
        "mu="
        f"{fingertip.parameters.optical.extinction_coefficient_m_inv:g} 1/m"
    )
    print()
    for index, label in enumerate(labels):
        print(f"{label}")
        print(f"  H (12 x 4):\n{response_array[index]}")
        print(f"  intensity (12D): {intensity[index]}")
        print(f"  spatial (4D):    {spatial[index]}")
    print()
    print(
        f"J_intensity={objective_intensity:.9e} | pair="
        f"({labels[intensity_pair[0]]}, {labels[intensity_pair[1]]})"
    )
    print(
        f"J_spatial={objective_spatial:.9e} | pair="
        f"({labels[spatial_pair[0]]}, {labels[spatial_pair[1]]})"
    )
    print("contact forces:")
    for trial in trials:
        print(
            f"  {trial.name}: F={trial.reaction_force_n:.4f} N | "
            f"travel={1.0e3 * trial.travel_m:.4f} mm"
        )
    print(f"wall runtime: {perf_counter() - wall_start:.3f} s")
    print("Side-view sensing evaluator: PASS")


if __name__ == "__main__":
    main()
