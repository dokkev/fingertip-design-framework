"""Reproject saved full-finger Newton states into the +X longitudinal view."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import MAIN_Y_BOUNDS_MM, make_fingertip_5led_mesh
from lumo.ray_tracing import (
    LED,
    OptixScene,
    emit_from_stem_boundary,
    longitudinal_side_view_observation,
    source_inside_silicone,
    trace_bounded_paths,
)


_INPUT_PATH = Path(
    "output/validation/full_finger_raw_evaluator/nominal_full_finger_raw.npz"
)
_OUTPUT_DIRECTORY = Path("output/validation/full_finger_spatial_observation")
_OUTPUT_PATH = _OUTPUT_DIRECTORY / "spatial_response.npz"
_REPORT_PATH = _OUTPUT_DIRECTORY / "report.md"

_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK
_SAMPLE_SIDE_COUNT = 256
_MAX_BOUNCES = 24
_RNG_SEED = 20260823
_CARRIER_ALBEDO = 0.7
_BIN_COUNT = 11


def _energy(paths: object) -> np.ndarray:
    return np.array(
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


def main() -> None:
    if not _INPUT_PATH.is_file():
        raise FileNotFoundError(_INPUT_PATH)
    with np.load(_INPUT_PATH) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}

    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_5led_mesh(fingertip)
    reference_vertices_m = np.asarray(mesh.silicone.vertices, dtype=np.float32)
    if reference_vertices_m.shape != data["reference_vertices_m"].shape or not np.allclose(
        reference_vertices_m,
        data["reference_vertices_m"],
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("saved Newton states do not match the current full mesh")
    if not np.allclose(
        mesh.led_centers_m,
        data["led_centers_m"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("saved LED centers do not match the current geometry")

    scene = OptixScene(
        mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    leds = tuple(
        LED(
            position_W_m=center_m,
            normal_W=np.array((0.0, 0.0, -1.0)),
            parameters=fingertip.parameters.led,
        )
        for center_m in mesh.led_centers_m
    )
    coordinate = (
        np.arange(_SAMPLE_SIDE_COUNT, dtype=np.float64) + 0.5
    ) / _SAMPLE_SIDE_COUNT
    sample_u1, sample_u2 = np.meshgrid(coordinate, coordinate, indexing="ij")
    emissions = tuple(
        emit_from_stem_boundary(
            scene,
            led,
            sample_u1.ravel(),
            sample_u2.ravel(),
            carrier_mask=_CARRIER_MASK,
        )
        for led in leds
    )
    rng = np.random.default_rng(_RNG_SEED)
    sample_shape = (_MAX_BOUNCES, len(emissions[0]))
    dielectric_branch_u = rng.random(sample_shape)
    carrier_u1 = rng.random(sample_shape)
    carrier_u2 = rng.random(sample_shape)

    scenario_count = len(data["scenario_names"])
    force_count = len(data["force_targets_n"])
    no_contact_response = np.empty((len(leds), _BIN_COUNT), dtype=np.float64)
    no_contact_energy = np.empty_like(data["no_contact_energy"])
    no_contact_total_camera_power = np.empty(len(leds), dtype=np.float64)
    no_contact_outside_active_power = np.empty(len(leds), dtype=np.float64)
    response_matrix = np.empty(
        (scenario_count, force_count, len(leds), _BIN_COUNT),
        dtype=np.float64,
    )
    energy_matrix = np.empty_like(data["energy_matrix"])
    total_camera_power = np.empty(
        (scenario_count, force_count, len(leds)),
        dtype=np.float64,
    )
    outside_active_power = np.empty_like(total_camera_power)
    y_min_m, y_max_m = (1.0e-3 * value for value in MAIN_Y_BOUNDS_MM)

    def trace_state(
        vertices_m: np.ndarray,
        responses: np.ndarray,
        energies: np.ndarray,
        camera_power: np.ndarray,
        outside_power: np.ndarray,
        *,
        require_air_sources: bool,
    ) -> None:
        scene.update_silicone(vertices_m)
        for led_index, (led, emission) in enumerate(
            zip(leds, emissions, strict=True)
        ):
            inside_silicone = source_inside_silicone(
                scene,
                led,
                emission,
                silicone_mask=_SILICONE_MASK,
            )
            if require_air_sources and inside_silicone:
                raise RuntimeError(
                    f"unloaded LED {led_index + 1} is not inside its air recess"
                )
            optics = fingertip.parameters.optics
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
            response = longitudinal_side_view_observation(paths.escaped_rays)
            directions = paths.escaped_rays["direction_W"]
            origins = paths.escaped_rays["origin_W_m"]
            power = paths.escaped_rays["power"]
            camera_visible = directions[:, 0] > 0.0
            active = (
                camera_visible
                & (origins[:, 1] >= y_min_m)
                & (origins[:, 1] <= y_max_m)
            )
            expected_active_power = float(power[active].sum())
            if not np.isclose(
                response.sum(),
                expected_active_power,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError("longitudinal bins do not close active-side power")
            responses[led_index] = response
            energies[led_index] = _energy(paths)
            camera_power[led_index] = float(power[camera_visible].sum())
            outside_power[led_index] = (
                camera_power[led_index] - expected_active_power
            )
            del paths

    trace_start_s = perf_counter()
    trace_state(
        reference_vertices_m,
        no_contact_response,
        no_contact_energy,
        no_contact_total_camera_power,
        no_contact_outside_active_power,
        require_air_sources=True,
    )
    for scenario_index in range(scenario_count):
        for force_index in range(force_count):
            trace_state(
                data["silicone_vertices_m"][scenario_index, force_index],
                response_matrix[scenario_index, force_index],
                energy_matrix[scenario_index, force_index],
                total_camera_power[scenario_index, force_index],
                outside_active_power[scenario_index, force_index],
                require_air_sources=False,
            )
    trace_runtime_s = perf_counter() - trace_start_s

    if not np.allclose(no_contact_energy, data["no_contact_energy"], atol=1.0e-12):
        raise RuntimeError("no-contact optical ledger changed during reprojection")
    if not np.allclose(energy_matrix, data["energy_matrix"], atol=1.0e-12):
        raise RuntimeError("loaded optical ledgers changed during reprojection")
    if not np.all(np.isfinite(response_matrix)):
        raise RuntimeError("spatial response contains non-finite values")

    bin_edges_y_m = np.linspace(y_min_m, y_max_m, _BIN_COUNT + 1)
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _OUTPUT_PATH,
        scenario_names=data["scenario_names"],
        sphere_diameters_mm=data["sphere_diameters_mm"],
        contact_y_mm=data["contact_y_mm"],
        force_targets_n=data["force_targets_n"],
        bin_edges_y_m=bin_edges_y_m,
        no_contact_response=no_contact_response,
        response_matrix=response_matrix,
        no_contact_energy=no_contact_energy,
        energy_matrix=energy_matrix,
        no_contact_total_camera_power=no_contact_total_camera_power,
        no_contact_outside_active_power=no_contact_outside_active_power,
        total_camera_power=total_camera_power,
        outside_active_power=outside_active_power,
        trace_runtime_s=np.array(trace_runtime_s),
    )
    maximum_outside_fraction = float(
        np.max(outside_active_power / total_camera_power)
    )
    combined_outside_fraction = outside_active_power.sum(axis=2) / (
        total_camera_power.sum(axis=2)
    )
    maximum_combined_outside_fraction = float(combined_outside_fraction.max())
    no_contact_combined_outside_fraction = float(
        no_contact_outside_active_power.sum()
        / no_contact_total_camera_power.sum()
    )
    _REPORT_PATH.write_text(
        "\n".join(
            (
                "# Full-finger +X spatial observation",
                "",
                "Result: PASS",
                "",
                "- Newton rerun: no; saved silicone vertex checkpoints reused",
                "- camera-facing direction: +X",
                "- image coordinate: world Y",
                "- active ROI: Y=[-27.5,+27.5] mm",
                "- bins: 11 x 5 mm",
                f"- response shape: `{response_matrix.shape}`",
                f"- optical trace runtime: {trace_runtime_s:.3f} s",
                f"- maximum power fraction outside active ROI: "
                f"{maximum_outside_fraction:.6%} for one emitter",
                f"- maximum simultaneous five-LED power fraction outside ROI: "
                f"{maximum_combined_outside_fraction:.6%}",
                f"- no-contact simultaneous power fraction outside ROI: "
                f"{no_contact_combined_outside_fraction:.6%}",
                "- saved energy ledgers: reproduced within 1e-12",
                "- every bin vector closes its active-ROI +X power within 1e-12",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print("Full-finger +X spatial observation PASS")
    print(f"response shape: {response_matrix.shape}")
    print(f"trace runtime: {trace_runtime_s:.3f} s")
    print(f"artifact: {_OUTPUT_PATH}")
    print(f"report: {_REPORT_PATH}")


if __name__ == "__main__":
    main()
