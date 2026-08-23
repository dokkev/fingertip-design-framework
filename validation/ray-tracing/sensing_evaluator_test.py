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


def _make_led(
    fingertip: Fingertip,
    fingertip_mesh: FingertipMesh,
) -> LED:
    """Return the pad-facing point source at the stem-bottom center."""
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


def _source_inside_silicone(
    scene: OptixScene,
    led: LED,
    emission: np.ndarray,
    *,
    state_label: str,
) -> bool:
    initial_hits = scene.trace_closest(
        emission["origin_W_m"],
        emission["direction_W"],
        mask=_ALL_MASK,
    )
    if not np.all(initial_hits["hit"]) or np.any(
        initial_hits["instance_id"] != _SILICONE_INSTANCE_ID
    ):
        raise AssertionError(f"{state_label} has an obstructed source ray")

    origins = emission["origin_W_m"][:1]
    emission_direction = led.normal_W[None, :]
    silicone_hit = scene.trace_closest(
        origins,
        emission_direction,
        mask=_SILICONE_MASK,
    )[0]
    if not silicone_hit["hit"]:
        raise AssertionError(f"{state_label} invalidated the stem source path")
    normal_projection = float(np.dot(silicone_hit["normal_W"], led.normal_W))
    if abs(normal_projection) <= 1.0e-6:
        raise AssertionError(f"{state_label} has an ambiguous source interface")
    return normal_projection > 0.0


def _emit_from_stem_boundary(
    scene: OptixScene,
    led: LED,
    u1: np.ndarray,
    u2: np.ndarray,
) -> np.ndarray:
    """Emit from the OTK-safe pad side of the physical source point."""
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
        raise AssertionError("carrier probe did not find the stem boundary")
    hit_position = probe_origin[0] + carrier_hit["t"][0] * led.normal_W
    if not np.allclose(
        hit_position,
        led.position_W_m,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise AssertionError("carrier probe found the wrong source boundary")

    safe_origin = safe_secondary_origins(carrier_hit, direction)[0]
    emission = led.emit(u1, u2)
    emission["origin_W_m"] = safe_origin
    return emission


def _trace_observation(
    scene: OptixScene,
    fingertip: Fingertip,
    emission: np.ndarray,
    *,
    inside_silicone: bool,
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
        inside_silicone=inside_silicone,
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

    if np.any(escaped["bounce"] == 0):
        raise AssertionError("stem source escaped without a silicone hit")
    return side_view_observation(
        escaped,
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
                np.linalg.norm(
                    np.atleast_1d(descriptors[first] - descriptors[second])
                )
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
    source_inside_silicone = _source_inside_silicone(
        scene,
        led,
        emission,
        state_label="no_contact",
    )
    if not source_inside_silicone:
        raise AssertionError("reference stem source is not touching silicone")
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
            inside_silicone=source_inside_silicone,
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
        source_inside_silicone = _source_inside_silicone(
            scene,
            led,
            emission,
            state_label=trial.name,
        )
        if trial.final_tf is None:
            raise AssertionError(f"{trial.name} has no final indenter pose")
        sphere_top_z_m = float(np.asarray(trial.final_tf)[2]) + _SPHERE_RADIUS_M
        if led.position_W_m[2] <= sphere_top_z_m:
            raise AssertionError(f"{trial.name} sphere reached the source plane")
        labels.append(trial.name)
        responses.append(
            _trace_observation(
                scene,
                fingertip,
                emission,
                inside_silicone=source_inside_silicone,
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
    if response_array.shape != (4, 4):
        raise AssertionError("side-view response does not have shape (4, 4)")
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
        f"LED position W [m]: {led.position_W_m} | normal={led.normal_W} | "
        "stem-bottom center | extrusion-axis center"
    )
    print(
        f"one optical cell | rays={len(emission)} | bounce_cap={_BOUNCE_CAP} | "
        "OTK-safe pad-side source; load-induced gap resolved"
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
        print(f"  quadrant power [Q1 Q2 Q3 Q4]: {response_array[index]}")
        print(f"  normalized intensity response: {intensity[index]:+.9e}")
        print(f"  normalized spatial response:   {spatial[index]}")
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
