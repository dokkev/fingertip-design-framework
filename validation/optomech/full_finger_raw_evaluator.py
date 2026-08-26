"""Validate one full-finger raw Newton-to-OptiX evaluation artifact."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.evaluator import FullFingerEvaluation, evaluate_full_finger
from lumo.ray_tracing import LONGITUDINAL_SIDE_BIN_COUNT


_OUTPUT_DIRECTORY = Path("output/validation/full_finger_raw_evaluator")
_ARTIFACT_PATH = _OUTPUT_DIRECTORY / "nominal_full_finger_raw.npz"
_REPORT_PATH = _OUTPUT_DIRECTORY / "report.md"
_SPHERE_DIAMETER_MM = 15.0
_CONTACT_Y_MM = (0.0, 5.5, 22.0)
_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
_SETTLE_DURATION_S = 5.0
_FORCE_TOLERANCE_FRACTION = 0.10


def _six_tet_volumes(
    positions_m: np.ndarray,
    tet_indices: np.ndarray,
) -> np.ndarray:
    tetrahedra = positions_m[tet_indices]
    return np.einsum(
        "ij,ij->i",
        tetrahedra[:, 1] - tetrahedra[:, 0],
        np.cross(
            tetrahedra[:, 2] - tetrahedra[:, 0],
            tetrahedra[:, 3] - tetrahedra[:, 0],
        ),
    )


def _save(evaluation: FullFingerEvaluation) -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _ARTIFACT_PATH,
        reference_vertices_m=evaluation.reference_vertices_m,
        tet_indices=evaluation.tet_indices,
        surface_triangles=evaluation.surface_triangles,
        bonded_vertex_indices=evaluation.bonded_vertex_indices,
        led_centers_m=evaluation.led_centers_m,
        no_contact_response=evaluation.no_contact_response,
        no_contact_energy=evaluation.no_contact_energy,
        no_contact_inside_roi_power=evaluation.no_contact_inside_roi_power,
        no_contact_outside_roi_power=evaluation.no_contact_outside_roi_power,
        no_contact_visible_side_power=evaluation.no_contact_visible_side_power,
        no_contact_outside_roi_power_fraction=np.asarray(
            evaluation.no_contact_outside_roi_power_fraction
        ),
        scenario_names=np.asarray(evaluation.scenario_names),
        sphere_diameters_mm=evaluation.sphere_diameters_mm,
        contact_y_mm=evaluation.contact_y_mm,
        force_targets_n=evaluation.force_targets_n,
        actual_forces_n=evaluation.actual_forces_n,
        indentations_m=evaluation.indentations_m,
        checkpoint_times_s=evaluation.checkpoint_times_s,
        maximum_particle_speeds_m_s=evaluation.maximum_particle_speeds_m_s,
        indenter_contact_counts=evaluation.indenter_contact_counts,
        total_contact_counts=evaluation.total_contact_counts,
        contact_buffer_overflow=evaluation.contact_buffer_overflow,
        minimum_det_f=evaluation.minimum_det_f,
        inverted_tet_counts=evaluation.inverted_tet_counts,
        contact_centroids_W_m=evaluation.contact_centroids_W_m,
        contact_record_offsets=evaluation.contact_record_offsets,
        contact_particle_indices=evaluation.contact_particle_indices,
        contact_barycentric=evaluation.contact_barycentric,
        contact_positions_W_m=evaluation.contact_positions_W_m,
        contact_normals_W=evaluation.contact_normals_W,
        contact_body_positions=evaluation.contact_body_positions,
        silicone_vertices_m=evaluation.silicone_vertices_m,
        response_matrix=evaluation.response_matrix,
        energy_fields=np.asarray(evaluation.energy_fields),
        energy_matrix=evaluation.energy_matrix,
        inside_roi_power=evaluation.inside_roi_power,
        outside_roi_power=evaluation.outside_roi_power,
        visible_side_power=evaluation.visible_side_power,
        outside_roi_power_fraction=evaluation.outside_roi_power_fraction,
        scenario_runtime_s=evaluation.scenario_runtime_s,
    )


def _reload_and_verify() -> dict[str, float | int]:
    with np.load(_ARTIFACT_PATH) as saved:
        arrays = {name: np.asarray(saved[name]) for name in saved.files}

    scenario_count = len(arrays["scenario_names"])
    force_count = len(arrays["force_targets_n"])
    if arrays["response_matrix"].shape != (
        scenario_count,
        force_count,
        5,
        LONGITUDINAL_SIDE_BIN_COUNT,
    ):
        raise RuntimeError("per-emitter response matrix has the wrong shape")
    if arrays["no_contact_response"].shape != (
        5,
        LONGITUDINAL_SIDE_BIN_COUNT,
    ):
        raise RuntimeError("no-contact response does not retain five emitters")
    if not np.all(np.isfinite(arrays["response_matrix"])):
        raise RuntimeError("reloaded optical responses are non-finite")
    combined_response = arrays["response_matrix"].sum(axis=2)
    if combined_response.shape != (
        scenario_count,
        force_count,
        LONGITUDINAL_SIDE_BIN_COUNT,
    ):
        raise RuntimeError("combined response derivation failed")

    energy_fields = tuple(str(value) for value in arrays["energy_fields"])
    closure_index = energy_fields.index("closure_error")
    maximum_closure_error = float(
        np.max(np.abs(arrays["energy_matrix"][..., closure_index]))
    )
    maximum_closure_error = max(
        maximum_closure_error,
        float(np.max(np.abs(arrays["no_contact_energy"][:, closure_index]))),
    )
    if maximum_closure_error > 1.0e-12:
        raise RuntimeError("reloaded optical energy ledger does not close")
    if not np.allclose(
        arrays["inside_roi_power"] + arrays["outside_roi_power"],
        arrays["visible_side_power"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("reloaded checkpoint ROI accounting does not close")
    if not np.allclose(
        arrays["no_contact_inside_roi_power"]
        + arrays["no_contact_outside_roi_power"],
        arrays["no_contact_visible_side_power"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("reloaded no-contact ROI accounting does not close")

    reference_volumes = _six_tet_volumes(
        arrays["reference_vertices_m"],
        arrays["tet_indices"],
    )
    offsets = arrays["contact_record_offsets"]
    positions = arrays["contact_positions_W_m"]
    for scenario_index in range(scenario_count):
        if np.any(np.diff(arrays["indentations_m"][scenario_index]) <= 0.0):
            raise RuntimeError("indentation is not increasing with target force")
        for force_index, target_force_n in enumerate(arrays["force_targets_n"]):
            actual_force_n = arrays["actual_forces_n"][scenario_index, force_index]
            if abs(actual_force_n - target_force_n) > (
                _FORCE_TOLERANCE_FRACTION * target_force_n
            ):
                raise RuntimeError("accepted force lies outside its tolerance band")

            start, count = offsets[scenario_index, force_index]
            contact_slice = positions[start : start + count]
            if count <= 0 or len(contact_slice) != count:
                raise RuntimeError("contact record offsets are invalid")
            if not np.allclose(
                contact_slice.mean(axis=0),
                arrays["contact_centroids_W_m"][scenario_index, force_index],
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError("contact centroid cannot be reconstructed")
            if count != arrays["indenter_contact_counts"][
                scenario_index,
                force_index,
            ]:
                raise RuntimeError("raw and scalar indenter contact counts differ")

            current_volumes = _six_tet_volumes(
                arrays["silicone_vertices_m"][scenario_index, force_index],
                arrays["tet_indices"],
            )
            det_f = current_volumes / reference_volumes
            if not np.isclose(
                det_f.min(),
                arrays["minimum_det_f"][scenario_index, force_index],
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise RuntimeError("minimum det(F) cannot be reconstructed")
            if np.count_nonzero(det_f <= 0.0) != arrays["inverted_tet_counts"][
                scenario_index,
                force_index,
            ]:
                raise RuntimeError("inversion count cannot be reconstructed")

    particle_indices = arrays["contact_particle_indices"]
    present = particle_indices >= 0
    if np.any(particle_indices[present] >= len(arrays["reference_vertices_m"])):
        raise RuntimeError("raw contacts contain an invalid local vertex index")
    if np.any(arrays["contact_buffer_overflow"] != 0):
        raise RuntimeError("a checkpoint overflowed the contact buffer")

    return {
        "scenario_count": scenario_count,
        "force_count": force_count,
        "contact_record_count": len(positions),
        "maximum_energy_closure_error": maximum_closure_error,
        "artifact_size_bytes": _ARTIFACT_PATH.stat().st_size,
    }


def _write_report(
    evaluation: FullFingerEvaluation,
    verification: dict[str, float | int],
    wall_runtime_s: float,
) -> None:
    lines = [
        "# Full-finger raw evaluator validation",
        "",
        "Result: PASS",
        "",
        f"- sphere diameter: {_SPHERE_DIAMETER_MM:g} mm",
        f"- contact Y locations: {list(_CONTACT_Y_MM)} mm",
        f"- force targets: {list(_FORCE_TARGETS_N)} N",
        f"- response shape: `{evaluation.response_matrix.shape}`",
        f"- no-contact response shape: `{evaluation.no_contact_response.shape}`",
        f"- contact records: {verification['contact_record_count']}",
        f"- artifact size: {verification['artifact_size_bytes']} bytes",
        f"- wall runtime: {wall_runtime_s:.3f} s",
        f"- maximum energy closure error: {verification['maximum_energy_closure_error']:.3e}",
        "",
        "## Checkpoints",
        "",
        "| scenario | target [N] | actual [N] | indentation [mm] | contacts | min det(F) | inverted | vmax [m/s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario_index, scenario_name in enumerate(evaluation.scenario_names):
        for force_index, target_force_n in enumerate(evaluation.force_targets_n):
            lines.append(
                f"| {scenario_name} | {target_force_n:.1f} | "
                f"{evaluation.actual_forces_n[scenario_index, force_index]:.6f} | "
                f"{1.0e3 * evaluation.indentations_m[scenario_index, force_index]:.6f} | "
                f"{evaluation.indenter_contact_counts[scenario_index, force_index]} | "
                f"{evaluation.minimum_det_f[scenario_index, force_index]:.6f} | "
                f"{evaluation.inverted_tet_counts[scenario_index, force_index]} | "
                f"{evaluation.maximum_particle_speeds_m_s[scenario_index, force_index]:.6e} |"
            )
    lines.extend(
        (
            "",
            "The saved NPZ was reloaded without a live Newton or OptiX runtime. "
            "Contact centroids, combined five-emitter longitudinal responses, minimum "
            "det(F), inversion counts, and energy closure were recomputed from "
            "the artifact.",
        )
    )
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    wall_start_s = perf_counter()
    fingertip = Fingertip(FingertipParameters())
    sphere_resource = files("lumo").joinpath(
        "assets",
        "objects",
        "urdf",
        "sphere_15mm.urdf",
    )
    with as_file(sphere_resource) as sphere_path:
        evaluation = evaluate_full_finger(
            fingertip,
            (sphere_path,),
            (_SPHERE_DIAMETER_MM,),
            _CONTACT_Y_MM,
            force_targets_n=_FORCE_TARGETS_N,
            settle_duration_s=_SETTLE_DURATION_S,
            force_tolerance_fraction=_FORCE_TOLERANCE_FRACTION,
        )
    _save(evaluation)
    verification = _reload_and_verify()
    wall_runtime_s = perf_counter() - wall_start_s
    _write_report(evaluation, verification, wall_runtime_s)

    print("Full-finger raw evaluator PASS")
    print(f"response shape: {evaluation.response_matrix.shape}")
    print(f"contact records: {verification['contact_record_count']}")
    print(f"artifact: {_ARTIFACT_PATH}")
    print(f"report: {_REPORT_PATH}")
    print(f"wall runtime: {wall_runtime_s:.3f} s")


if __name__ == "__main__":
    main()
