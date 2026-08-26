"""Benchmark accepted Newton reuse modes on the complete 21-scenario grid."""

from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from time import perf_counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.optimization.evaluator import evaluate_full_finger
from lumo.optimization.objective import compute_objectives_from_raw
import lumo.simulation.runtime as runtime_module


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = (
    _ROOT
    / "output"
    / "validation"
    / "production_evaluator_acceleration"
    / "phase2_runtime_reuse"
)
_SPHERES = (("sphere_5mm.urdf", 5.0), ("sphere_10mm.urdf", 10.0), ("sphere_20mm.urdf", 20.0))
_CONTACT_Y_MM = (-22.0, -11.0, -5.5, 0.0, 5.5, 11.0, 22.0)


def _run_mode(
    name: str,
    sphere_paths: tuple[Path, ...],
    *,
    reuse_finalized_models: bool,
    reuse_runtimes: bool,
) -> dict[str, np.ndarray]:
    output_path = _OUTPUT_DIRECTORY / f"full_grid_{name}.npz"
    if output_path.is_file():
        print(f"{name}: loading {output_path.name}", flush=True)
        with np.load(output_path) as stored:
            return {key: stored[key] for key in stored.files}

    build_count = 0
    build_wall_s = 0.0
    original_build = runtime_module.build_fingertip_newton_model

    def timed_build(*args: object, **kwargs: object) -> object:
        nonlocal build_count, build_wall_s
        start_s = perf_counter()
        built = original_build(*args, **kwargs)
        build_wall_s += perf_counter() - start_s
        build_count += 1
        return built

    runtime_module.build_fingertip_newton_model = timed_build
    print(f"{name}: running full 3 x 7 grid", flush=True)
    start_s = perf_counter()
    try:
        evaluation = evaluate_full_finger(
            Fingertip(FingertipParameters()),
            sphere_paths,
            tuple(diameter for _, diameter in _SPHERES),
            _CONTACT_Y_MM,
            reuse_finalized_models=reuse_finalized_models,
            reuse_runtimes=reuse_runtimes,
        )
    finally:
        runtime_module.build_fingertip_newton_model = original_build
    wall_s = perf_counter() - start_s
    contact, observation = compute_objectives_from_raw(vars(evaluation))
    arrays = {
        "wall_s": np.asarray(wall_s),
        "model_build_count": np.asarray(build_count),
        "model_build_wall_s": np.asarray(build_wall_s),
        "actual_forces_n": evaluation.actual_forces_n,
        "indentations_m": evaluation.indentations_m,
        "checkpoint_steps": evaluation.checkpoint_steps,
        "minimum_det_f": evaluation.minimum_det_f,
        "inverted_tet_counts": evaluation.inverted_tet_counts,
        "contact_buffer_overflow": evaluation.contact_buffer_overflow,
        "q_form": contact.q_form,
        "q_stable": contact.q_stable,
        "q_stiff": contact.q_stiff,
        "q_contact": contact.q_contact,
        "J_contact": np.asarray(contact.J_contact),
        "response_matrix": evaluation.response_matrix,
        "location_separations": observation.location_separations,
        "J_obs": np.asarray(observation.J_obs),
        "mechanics_backend": np.asarray(evaluation.mechanics_backend),
    }
    np.savez_compressed(output_path, **arrays)
    print(f"{name}: {wall_s:.3f} s", flush=True)
    return arrays


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(candidate - reference) / scale))


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    resource_root = files("lumo.assets.objects.urdf")
    with ExitStack() as resources:
        sphere_paths = tuple(
            resources.enter_context(as_file(resource_root.joinpath(filename)))
            for filename, _ in _SPHERES
        )
        modes = {
            "fresh": _run_mode(
                "fresh",
                sphere_paths,
                reuse_finalized_models=False,
                reuse_runtimes=False,
            ),
            "stage_a": _run_mode(
                "stage_a",
                sphere_paths,
                reuse_finalized_models=True,
                reuse_runtimes=False,
            ),
            "stage_b": _run_mode(
                "stage_b",
                sphere_paths,
                reuse_finalized_models=True,
                reuse_runtimes=True,
            ),
        }

    reference = modes["fresh"]
    lines = [
        "# Newton reuse full-grid benchmark",
        "",
        "All modes use the accepted GPU-resident force-servo graph and the same "
        "finite-area optical evaluation over 3 sphere diameters x 7 Y locations.",
        "",
        "| mode | backend | model builds | build wall [s] | total wall [s] | speedup | J_contact | J_obs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in modes.items():
        wall_s = float(values["wall_s"])
        lines.append(
            f"| {name} | {str(values['mechanics_backend'])} | "
            f"{int(values['model_build_count'])} | "
            f"{float(values['model_build_wall_s']):.3f} | {wall_s:.3f} | "
            f"{float(reference['wall_s']) / wall_s:.3f}x | "
            f"{float(values['J_contact']):.9f} | {float(values['J_obs']):.9f} |"
        )
    lines.extend(("", "## Full-grid numerical comparison against fresh", ""))
    for name in ("stage_a", "stage_b"):
        values = modes[name]
        force_abs = float(
            np.max(np.abs(values["actual_forces_n"] - reference["actual_forces_n"]))
        )
        indentation_abs_um = 1.0e6 * float(
            np.max(np.abs(values["indentations_m"] - reference["indentations_m"]))
        )
        step_abs = int(
            np.max(np.abs(values["checkpoint_steps"] - reference["checkpoint_steps"]))
        )
        lines.extend(
            (
                f"### {name}",
                "",
                f"- maximum checkpoint-step difference: {step_abs}",
                f"- maximum force difference: {force_abs:.6f} N",
                f"- maximum indentation difference: {indentation_abs_um:.3f} um",
                f"- q_form maximum relative difference: {100.0 * _relative_error(values['q_form'], reference['q_form']):.3f}%",
                f"- q_stable maximum relative difference: {100.0 * _relative_error(values['q_stable'], reference['q_stable']):.3f}%",
                f"- q_stiff maximum relative difference: {100.0 * _relative_error(values['q_stiff'], reference['q_stiff']):.3f}%",
                f"- q_contact maximum relative difference: {100.0 * _relative_error(values['q_contact'], reference['q_contact']):.3f}%",
                f"- J_contact relative difference: {100.0 * _relative_error(values['J_contact'], reference['J_contact']):.3f}%",
                f"- 252 location separations maximum relative difference: {100.0 * _relative_error(values['location_separations'], reference['location_separations']):.3f}%",
                f"- J_obs relative difference: {100.0 * _relative_error(values['J_obs'], reference['J_obs']):.3f}%",
                f"- minimum det(F): {float(np.min(values['minimum_det_f'])):.6f}",
                f"- inversions: {int(np.max(values['inverted_tet_counts']))}",
                f"- contact-buffer overflow: {int(np.max(values['contact_buffer_overflow']))}",
                "",
            )
        )
    report = "\n".join(lines)
    (_OUTPUT_DIRECTORY / "full_grid_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
