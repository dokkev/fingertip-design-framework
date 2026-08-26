"""Run the final accelerated nominal full-finger production evaluation."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.evaluator import evaluate_full_finger
from lumo.optimization.objective import compute_objectives_from_raw


_OUTPUT_DIRECTORY = (
    Path("output/validation/production_evaluator_acceleration")
)
_RAW_PATH = _OUTPUT_DIRECTORY / "final_nominal.npz"
_REPORT_PATH = _OUTPUT_DIRECTORY / "report.md"
_CONTACT_Y_MM = (-22.0, -11.0, -5.5, 0.0, 5.5, 11.0, 22.0)
_SPHERES = (
    ("sphere_5mm.urdf", 5.0),
    ("sphere_10mm.urdf", 10.0),
    ("sphere_20mm.urdf", 20.0),
)
_ORIGINAL_WALL_S = 1222.991
_CAMPAIGN_COUNTS = (60, 100, 120)


def _run() -> tuple[dict[str, np.ndarray], float]:
    if _RAW_PATH.is_file():
        with np.load(_RAW_PATH) as saved:
            data = {name: np.asarray(saved[name]) for name in saved.files}
        return data, float(data["runtime_s"])

    resource_root = files("lumo.assets.objects.urdf")
    start_s = perf_counter()
    with (
        as_file(resource_root.joinpath(_SPHERES[0][0])) as sphere_5mm,
        as_file(resource_root.joinpath(_SPHERES[1][0])) as sphere_10mm,
        as_file(resource_root.joinpath(_SPHERES[2][0])) as sphere_20mm,
    ):
        evaluation = evaluate_full_finger(
            Fingertip(FingertipParameters()),
            (sphere_5mm, sphere_10mm, sphere_20mm),
            tuple(diameter for _, diameter in _SPHERES),
            _CONTACT_Y_MM,
            parallel_world_count=4,
        )
    runtime_s = perf_counter() - start_s
    contact, observation = compute_objectives_from_raw(vars(evaluation))
    data = {
        **{
            name: np.asarray(value)
            for name, value in vars(evaluation).items()
            if value is not None
        },
        "runtime_s": np.asarray(runtime_s),
        "J_contact": np.asarray(contact.J_contact),
        "J_obs": np.asarray(observation.J_obs),
        "contact_limiting_scenario": np.asarray(contact.limiting_scenario),
        "observation_limiting_sphere_diameter_mm": np.asarray(
            observation.limiting_sphere_diameter_mm
        ),
        "observation_limiting_force_n": np.asarray(
            observation.limiting_force_n
        ),
        "observation_limiting_contact_y_pair_mm": np.asarray(
            observation.limiting_contact_y_pair_mm
        ),
    }
    np.savez_compressed(_RAW_PATH, **data)
    return data, runtime_s


def _verify(data: dict[str, np.ndarray]) -> tuple[object, object, float]:
    if np.any(data["inverted_tet_counts"] != 0):
        raise RuntimeError("final nominal evaluation contains inverted tetrahedra")
    if np.any(data["contact_buffer_overflow"] != 0):
        raise RuntimeError("final nominal evaluation overflowed a contact buffer")
    if np.any(data["indenter_contact_counts"] <= 0):
        raise RuntimeError("final nominal evaluation contains a contact-free checkpoint")
    if not np.all(np.isfinite(data["silicone_vertices_m"])):
        raise RuntimeError("final nominal silicone state is non-finite")
    if not np.all(np.isfinite(data["response_matrix"])):
        raise RuntimeError("final nominal optical response is non-finite")
    if not np.allclose(
        data["response_matrix"].sum(axis=-1),
        data["inside_roi_power"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("11-bin optical power does not close")
    if not np.allclose(
        data["inside_roi_power"] + data["outside_roi_power"],
        data["visible_side_power"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("inside/outside ROI optical power does not close")
    closure_index = tuple(str(value) for value in data["energy_fields"]).index(
        "closure_error"
    )
    closure_error = float(
        max(
            np.max(np.abs(data["energy_matrix"][..., closure_index])),
            np.max(np.abs(data["no_contact_energy"][:, closure_index])),
        )
    )
    if closure_error > 1.0e-12:
        raise RuntimeError("final nominal optical energy ledger does not close")
    contact, observation = compute_objectives_from_raw(data)
    if not np.isclose(contact.J_contact, float(data["J_contact"])):
        raise RuntimeError("J_contact is not reproducible from final raw NPZ")
    if not np.isclose(observation.J_obs, float(data["J_obs"])):
        raise RuntimeError("J_obs is not reproducible from final raw NPZ")
    return contact, observation, closure_error


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data, runtime_s = _run()
    contact, observation, closure_error = _verify(data)
    speedup = _ORIGINAL_WALL_S / runtime_s
    phase2_path = _OUTPUT_DIRECTORY / "phase2_runtime_reuse" / "full_grid_stage_b.npz"
    phase1_wall_s = np.nan
    phase2_wall_s = np.nan
    if phase2_path.is_file():
        with np.load(phase2_path) as phase2:
            phase2_wall_s = float(phase2["wall_s"])
        fresh_path = phase2_path.with_name("full_grid_fresh.npz")
        with np.load(fresh_path) as phase1:
            phase1_wall_s = float(phase1["wall_s"])

    lines = [
        "# Production evaluator acceleration",
        "",
        "| Stage | Contract | Runtime / morphology | Speedup vs original | Numerical status | Production accepted? |",
        "|---|---|---:|---:|---|---|",
        f"| Original | point/hard, direct, 5 s, serial | {_ORIGINAL_WALL_S:.3f} s | 1.000x | reference | no |",
        f"| Phase 1 | finite-area/hard + GPU graph | {phase1_wall_s:.3f} s | {_ORIGINAL_WALL_S / phase1_wall_s:.3f}x | intrinsic-envelope PASS | yes |",
        f"| Phase 2 | + finalized model/runtime reuse | {phase2_wall_s:.3f} s | {_ORIGINAL_WALL_S / phase2_wall_s:.3f}x | reset/reuse PASS | yes |",
        f"| Phase 3 | + dwell screen | {phase2_wall_s:.3f} s | {_ORIGINAL_WALL_S / phase2_wall_s:.3f}x | shorter dwell FAIL; 5 s retained | yes |",
        f"| Phase 4 | + 4 parallel worlds | {runtime_s:.3f} s | {speedup:.3f}x | full nominal E2E PASS | yes |",
        "",
        "## Final nominal result",
        "",
        f"- mechanics backend: {str(data['mechanics_backend'])}",
        f"- J_contact: {contact.J_contact:.9f} ({contact.limiting_scenario})",
        f"- J_obs: {observation.J_obs:.9f} (sphere {observation.limiting_sphere_diameter_mm:g} mm, {observation.limiting_force_n:g} N, Y={observation.limiting_contact_y_pair_mm})",
        f"- minimum det(F): {float(np.min(data['minimum_det_f'])):.6f}",
        "- inversion / contact-buffer overflow: 0 / 0",
        f"- maximum outside-ROI fraction: {float(np.max(data['outside_roi_power_fraction'])):.6%}",
        f"- maximum energy closure error: {closure_error:.3e}",
        "- objective recomputation from final_nominal.npz: PASS",
        "",
        "## Campaign-time projection",
        "",
    ]
    for count in _CAMPAIGN_COUNTS:
        total_s = count * runtime_s
        lines.append(
            f"- {count} morphologies: {total_s / 3600.0:.2f} h "
            f"({total_s / 86400.0:.2f} d)"
        )
    lines.extend(
        (
            "",
            "## Contract classification",
            "",
            "- Scientific correction: finite 1.8 x 1.6 mm hardware-informed LED aperture with hard 11-bin observation.",
            "- Implementation-only acceleration: GPU-resident servo/graph, model/runtime reuse, and four independent CUDA-stream worlds.",
            "- Dwell: no acceleration accepted; the conservative 5 s force-band dwell remains production.",
            "- No production BO campaign was started.",
        )
    )
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(_REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
