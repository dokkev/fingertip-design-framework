"""Timed, prescribed-indentation Newton VBD benchmark on the nominal fingertip."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from lumo.physics import NewtonSettings, prepare_fingertip_mesh
from lumo.physics.trajectory.fingertip_adapter import (
    outer_compliant_timing_patch,
    solve_prescribed_indentation,
)
from lumo.mesh.volume.mesh import generate_volume_mesh
from lumo.mesh.volume.contracts import volume_mesh_settings_for_tier
from lumo.finger.fingertip_geometry import FingertipModel
from lumo.finger.fingertip_parameters import FingertipParameters
from lumo.finger.extrusion import build_fingertip_solid
from tests.validation.common.io import atomic_write_json


def _package_version(distribution: str, module: Any) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exception:
        return {"revision": None, "working_tree_dirty": None, "error": str(exception)}
    return {"revision": revision, "working_tree_dirty": bool(dirty)}


def _timed_run(prepared, settings, patch):
    return solve_prescribed_indentation(prepared, settings, patch)


def run_nominal_benchmark(
    *,
    repo_root: str | Path = ".",
    warm_repeats: int = 3,
) -> dict[str, Any]:
    """Run one cold, one untimed warm-up, and repeated warm VBD solves."""

    if warm_repeats < 1:
        raise ValueError("warm_repeats must be positive")
    root = Path(repo_root).resolve()
    preprocessing_start = time.perf_counter()
    parameters = FingertipParameters()
    model = FingertipModel(parameters)
    solid = build_fingertip_solid(model)
    mesh_settings = volume_mesh_settings_for_tier("search")
    volume_mesh = generate_volume_mesh(solid, mesh_settings)
    prepared = prepare_fingertip_mesh(volume_mesh)
    preprocessing_wall_s = time.perf_counter() - preprocessing_start

    patch = outer_compliant_timing_patch(
        prepared,
        displacement_mm=(0.0, 0.5, 0.0),
        load_steps=8,
    )
    settings = NewtonSettings(
        device="cuda:0",
        gravity=0.0,
        dt=1.0 / 60.0,
        steps=patch.load_steps,
        iterations=5,
        fixed_vertex_indices=prepared.support_vertex_indices,
    )

    # The first invocation is intentionally retained as the cold/first-process
    # observation.  The following invocation is an untimed warm-up, after
    # which each repeated call rebuilds and resets the same model state.
    first_result, first_timing = _timed_run(prepared, settings, patch)
    _timed_run(prepared, settings, patch)
    warm_timings: list[dict[str, Any]] = []
    for _ in range(warm_repeats):
        result, timing = _timed_run(prepared, settings, patch)
        warm_timings.append(timing)

    selected = np.asarray(patch.vertex_indices, dtype=np.int64)
    target = np.asarray(patch.displacement_mm, dtype=np.float32)
    achieved_error_mm = float(
        np.max(np.abs(first_result.displacement[selected] - target))
    )

    import newton
    import warp as wp

    device = wp.get_device(settings.device)
    warm_solver = [float(item["solver_loop_wall_s"]) for item in warm_timings]
    warm_build = [float(item["model_build_wall_s"]) for item in warm_timings]
    warm_total = [float(item["total_mechanics_wall_s"]) for item in warm_timings]
    return {
        "schema": "physics-vbd-prescribed-indentation-timing-v1",
        "case": "nominal_prescribed_0p5mm",
        "scientific_role": "timing-only prescribed indentation patch; not rigid-indenter contact",
        "fea_rerun": False,
        "optix_run": False,
        "mesh_tier": "search",
        "morphology_fingerprint": prepared.morphology_fingerprint,
        "node_count": int(prepared.tet_mesh.vertices.shape[0]),
        "tet_count": int(prepared.tet_mesh.tetrahedra.shape[0]),
        "device": settings.device,
        "device_name": str(getattr(device, "name", device)),
        "warp_version": _package_version("warp-lang", wp),
        "newton_version": _package_version("newton", newton),
        "fixed_vertex_count": len(prepared.support_vertex_indices),
        "prescribed_vertex_count": len(patch.vertex_indices),
        "prescribed_patch_label": patch.label,
        "prescribed_displacement_mm": list(patch.displacement_mm),
        "prescribed_indentation_mm": 0.5,
        "load_steps": patch.load_steps,
        "iterations_per_step": settings.iterations,
        "dt": settings.dt,
        "gravity": settings.gravity,
        "preprocessing_wall_s": preprocessing_wall_s,
        "first_run": first_timing,
        "untimed_warmup": True,
        "warm_repeats": warm_repeats,
        "warm_model_build_median_s": float(np.median(warm_build)),
        "warm_model_build_min_s": float(np.min(warm_build)),
        "warm_model_build_max_s": float(np.max(warm_build)),
        "warm_solver_loop_median_s": float(np.median(warm_solver)),
        "warm_solver_loop_min_s": float(np.min(warm_solver)),
        "warm_solver_loop_max_s": float(np.max(warm_solver)),
        "warm_total_mechanics_median_s": float(np.median(warm_total)),
        "warm_total_mechanics_min_s": float(np.min(warm_total)),
        "warm_total_mechanics_max_s": float(np.max(warm_total)),
        "prescribed_target_max_error_mm": achieved_error_mm,
        "finite_result": bool(
            np.all(np.isfinite(first_result.deformed_vertices))
            and np.all(np.isfinite(first_result.displacement))
        ),
        "git": _git_provenance(root),
        "timing_note": "GPU work is synchronized before/after model construction and solver loops; initialization is outside total mechanics timing.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/validation/physics/vbd_nominal_indentation_timing.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--warm-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = run_nominal_benchmark(repo_root=args.repo_root, warm_repeats=args.warm_repeats)
    except Exception as exception:
        print(f"FAIL: prescribed_vbd_benchmark: {exception}")
        return 1
    atomic_write_json(args.output, result)
    print(
        "PASS: prescribed VBD nominal 0.5 mm "
        f"nodes={result['node_count']} tets={result['tet_count']} "
        f"build={result['first_run']['model_build_wall_s']:.6f}s "
        f"solve={result['first_run']['solver_loop_wall_s']:.6f}s "
        f"total={result['first_run']['total_mechanics_wall_s']:.6f}s"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_nominal_benchmark"]
