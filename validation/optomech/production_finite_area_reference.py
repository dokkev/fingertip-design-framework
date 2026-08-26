"""Replay frozen Newton states with the production finite-area LED source."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_mesh
from lumo.optimization.evaluator import (
    _CARRIER_INSTANCE_ID,
    _CARRIER_MASK,
    _SILICONE_INSTANCE_ID,
    _SILICONE_MASK,
    _full_finger_emissions,
    _full_finger_optical_samples,
    _make_full_finger_leds,
    _trace_full_finger_state,
)
from lumo.optimization.objective import compute_objectives_from_raw
from lumo.ray_tracing import OptixScene


_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = (
    _ROOT
    / "output"
    / "validation"
    / "full_finger_production_objective_freeze"
    / "nominal_full_finger_objectives.npz"
)
_OUTPUT_DIRECTORY = (
    _ROOT
    / "output"
    / "validation"
    / "production_evaluator_acceleration"
    / "phase1_cuda_graph"
)
_OUTPUT_PATH = _OUTPUT_DIRECTORY / "finite_area_reference.npz"
_REPORT_PATH = _OUTPUT_DIRECTORY / "finite_area_reference.md"


def main() -> None:
    if not _REFERENCE_PATH.is_file():
        raise FileNotFoundError(_REFERENCE_PATH)
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with np.load(_REFERENCE_PATH) as saved:
        reference = {name: saved[name] for name in saved.files}

    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_mesh(fingertip)
    if not np.allclose(
        mesh.silicone.vertices,
        reference["reference_vertices_m"],
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("saved Newton states do not match the current mesh")
    scene = OptixScene(
        mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    leds = _make_full_finger_leds(fingertip, mesh)
    emissions = _full_finger_emissions(scene, leds)
    branch_u, carrier_u1, carrier_u2 = _full_finger_optical_samples(
        len(emissions[0])
    )

    scenario_count, force_count = reference["actual_forces_n"].shape
    response = np.empty((scenario_count, force_count, 5, 11), dtype=np.float64)
    energy = np.empty((scenario_count, force_count, 5, 8), dtype=np.float64)
    outside = np.empty((scenario_count, force_count, 5), dtype=np.float64)
    visible = np.empty_like(outside)
    trace_start_s = perf_counter()
    scene.update_silicone(reference["reference_vertices_m"])
    no_contact_response, no_contact_energy, no_contact_outside, no_contact_visible = (
        _trace_full_finger_state(
            scene,
            fingertip,
            leds,
            emissions,
            dielectric_branch_u=branch_u,
            carrier_u1=carrier_u1,
            carrier_u2=carrier_u2,
            require_air_sources=True,
        )
    )
    for scenario_index in range(scenario_count):
        for force_index in range(force_count):
            scene.update_silicone(
                reference["silicone_vertices_m"][scenario_index, force_index]
            )
            (
                response[scenario_index, force_index],
                energy[scenario_index, force_index],
                outside[scenario_index, force_index],
                visible[scenario_index, force_index],
            ) = _trace_full_finger_state(
                scene,
                fingertip,
                leds,
                emissions,
                dielectric_branch_u=branch_u,
                carrier_u1=carrier_u1,
                carrier_u2=carrier_u2,
            )
        print(
            f"finite-area optics: {scenario_index + 1}/{scenario_count} scenarios",
            flush=True,
        )
    runtime_s = perf_counter() - trace_start_s

    objective_data = {
        **reference,
        "response_matrix": response,
        "no_contact_response": no_contact_response,
        "energy_matrix": energy,
        "no_contact_energy": no_contact_energy,
    }
    contact, observation = compute_objectives_from_raw(objective_data)
    combined_visible = visible.sum(axis=2)
    outside_fraction = np.divide(
        outside.sum(axis=2),
        combined_visible,
        out=np.zeros_like(combined_visible),
        where=combined_visible > 0.0,
    )
    no_contact_outside_fraction = float(
        no_contact_outside.sum() / no_contact_visible.sum()
    )
    maximum_closure_error = float(
        max(
            np.max(np.abs(energy[..., -1])),
            np.max(np.abs(no_contact_energy[..., -1])),
        )
    )
    np.savez_compressed(
        _OUTPUT_PATH,
        scenario_names=reference["scenario_names"],
        sphere_diameters_mm=reference["sphere_diameters_mm"],
        contact_y_mm=reference["contact_y_mm"],
        force_targets_n=reference["force_targets_n"],
        no_contact_response=no_contact_response,
        no_contact_energy=no_contact_energy,
        no_contact_outside_roi_power=no_contact_outside,
        no_contact_visible_side_power=no_contact_visible,
        response_matrix=response,
        energy_matrix=energy,
        outside_roi_power=outside,
        visible_side_power=visible,
        outside_roi_power_fraction=outside_fraction,
        J_contact=np.asarray(contact.J_contact),
        J_obs=np.asarray(observation.J_obs),
        limiting_sphere_diameter_mm=np.asarray(
            observation.limiting_sphere_diameter_mm
        ),
        limiting_force_n=np.asarray(observation.limiting_force_n),
        limiting_contact_y_pair_mm=np.asarray(
            observation.limiting_contact_y_pair_mm
        ),
        optical_runtime_s=np.asarray(runtime_s),
    )
    _REPORT_PATH.write_text(
        "\n".join(
            (
                "# Production finite-area optical reference",
                "",
                "Frozen Newton 5 s dwell states were replayed; Newton was not rerun.",
                "",
                "- source: uniform 1.8 mm x 1.6 mm LuckyLight resin window",
                "- binning: hard 11 x 5 mm longitudinal bins",
                "- emitted paths: 65,536 per LED, five simultaneous LEDs",
                f"- J_contact (unchanged): {contact.J_contact:.9f}",
                f"- J_obs: {observation.J_obs:.9f}",
                "- limiting observation: "
                f"sphere {observation.limiting_sphere_diameter_mm:g} mm, "
                f"{observation.limiting_force_n:g} N, "
                f"Y={observation.limiting_contact_y_pair_mm[0]:+g} vs "
                f"{observation.limiting_contact_y_pair_mm[1]:+g} mm",
                f"- no-contact outside-ROI fraction: {no_contact_outside_fraction:.6%}",
                f"- maximum loaded outside-ROI fraction: {outside_fraction.max():.6%}",
                f"- maximum energy closure error: {maximum_closure_error:.3e}",
                f"- optical replay runtime: {runtime_s:.3f} s",
                "",
            )
        )
    )
    print(_REPORT_PATH.read_text(), end="")
    print(f"Saved {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
