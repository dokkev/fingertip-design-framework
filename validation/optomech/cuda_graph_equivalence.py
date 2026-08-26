"""Validate and benchmark the partial Newton CUDA-step graphs."""

from __future__ import annotations

import csv
import re
from collections import Counter
from importlib.resources import as_file, files
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import numpy as np
import warp as wp

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_5led_mesh
from lumo.newton import Indenter
from lumo.optimization.evaluator import (
    _indenter_contact_records,
    _six_tet_volumes,
)
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = _ROOT / "output" / "validation" / "cuda_graph_equivalence"
_FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
_SIM_FREQUENCY_HZ = 100.0
_VBD_ITERATIONS = 10
_SETTLE_DURATION_S = 5.0
_FORCE_TOLERANCE_FRACTION = 0.1
_FORCE_GAIN_M_S_N = 2.5e-4
_APPROACH_SPEED_M_S = 5.0e-3
_CONTACT_STIFFNESS_N_M = 3.0e4
_CONTACT_DAMPING_N_S_M = 0.28228017516945547
_CARRIER_CONTACT_STIFFNESS_N_M = 1.0e6
_SOFT_CONTACT_MARGIN_M = 1.0e-4
_INITIAL_CLEARANCE_M = 1.0e-3
_MAX_SIM_TIME_S = 60.0
_EVENT_SAMPLE_STRIDE = 10
_DETAILED_GPU_WINDOW_STEPS = 20


def _new_profile(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "simulation": None,
        "step_wall_s": 0.0,
        "reaction_readback_wall_s": 0.0,
        "force_trajectory_n": [],
        "step_events": [],
        "arm_detailed_window": False,
        "detailed_window_started": False,
        "detailed_steps_remaining": 0,
        "detailed_results": [],
        "detailed_events": None,
        "checkpoints": [],
        "loop_start_s": None,
        "loop_end_s": None,
        "graph_replays": 0,
        "graph_nodes": {},
    }


def _install_instrumentation(
    profile: dict[str, Any],
) -> list[tuple[object, str, object]]:
    originals: list[tuple[object, str, object]] = []

    def patch(owner: object, name: str, replacement: object) -> None:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    original_init = LumoSimulation.__init__

    def simulation_init(
        self: LumoSimulation,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_init(self, *args, **kwargs)
        profile["simulation"] = self

    patch(LumoSimulation, "__init__", simulation_init)

    original_step = LumoSimulation.step

    def simulation_step(self: LumoSimulation) -> None:
        if profile["loop_start_s"] is None:
            profile["loop_start_s"] = perf_counter()
        if profile["arm_detailed_window"]:
            wp.timing_begin(cuda_filter=wp.TIMING_ALL, synchronize=True)
            events = (
                wp.Event(device=self.fingertip_model.model.device, enable_timing=True),
                wp.Event(device=self.fingertip_model.model.device, enable_timing=True),
            )
            wp.record_event(events[0])
            profile["detailed_events"] = events
            profile["detailed_steps_remaining"] = _DETAILED_GPU_WINDOW_STEPS
            profile["detailed_window_started"] = True
            profile["arm_detailed_window"] = False

        ordinal = self.step_count + 1
        sample = ordinal > 1 and ordinal % _EVENT_SAMPLE_STRIDE == 0
        events = None
        if sample:
            events = (
                wp.Event(device=self.fingertip_model.model.device, enable_timing=True),
                wp.Event(device=self.fingertip_model.model.device, enable_timing=True),
            )
            wp.record_event(events[0])
        start_s = perf_counter()
        original_step(self)
        profile["step_wall_s"] += perf_counter() - start_s
        if events is not None:
            wp.record_event(events[1])
            profile["step_events"].append(events)

        if profile["detailed_steps_remaining"]:
            profile["detailed_steps_remaining"] -= 1
            if profile["detailed_steps_remaining"] == 0:
                wp.record_event(profile["detailed_events"][1])
                profile["detailed_results"] = wp.timing_end(synchronize=True)

    patch(LumoSimulation, "step", simulation_step)

    original_reaction_force = LumoSimulation.indenter_reaction_force

    def reaction_force(
        self: LumoSimulation,
        *args: object,
        **kwargs: object,
    ) -> float:
        start_s = perf_counter()
        value = original_reaction_force(self, *args, **kwargs)
        profile["reaction_readback_wall_s"] += perf_counter() - start_s
        profile["force_trajectory_n"].append(value)
        if value > 1.0 and not profile["detailed_window_started"]:
            profile["arm_detailed_window"] = True
        return value

    patch(LumoSimulation, "indenter_reaction_force", reaction_force)
    return originals


def _restore_instrumentation(
    originals: list[tuple[object, str, object]],
) -> None:
    for owner, name, original in reversed(originals):
        setattr(owner, name, original)


def _graph_node_counts(path: Path) -> dict[str, int]:
    text = path.read_text()
    return {
        "total": len(re.findall(r'\[.*shape="record"', text)),
        "kernel": text.count('label="{KERNEL'),
        "memcpy": text.count('label="{MEMCPY'),
        "memset": text.count('label="{MEMSET'),
    }


def _capture_graph_topology(
    profile: dict[str, Any],
    simulation: LumoSimulation,
) -> None:
    if simulation._step_graph_ab is None or simulation._step_graph_ba is None:
        return
    for direction, graph in (
        ("AB", simulation._step_graph_ab),
        ("BA", simulation._step_graph_ba),
    ):
        path = _OUTPUT_DIRECTORY / f"graph_{direction}.dot"
        wp.capture_debug_dot_print(graph, str(path), verbose=True)
        profile["graph_nodes"][direction] = _graph_node_counts(path)


def _run_scenario(
    *,
    use_cuda_graph: bool,
    fingertip: Fingertip,
    fingertip_mesh: object,
    sphere_urdf_path: Path,
    tet_indices: np.ndarray,
    reference_six_volumes_m3: np.ndarray,
) -> dict[str, Any]:
    profile = _new_profile("graph" if use_cuda_graph else "direct")

    def collect_checkpoint(
        completed_trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        vertices_m = simulation.silicone_vertices()
        records = _indenter_contact_records(simulation, indenter, vertices_m)
        det_f = (
            _six_tet_volumes(vertices_m, tet_indices)
            / reference_six_volumes_m3
        )
        overflow = int(
            simulation.solver.body_particle_contact_overflow_max.numpy()[0]
        )
        profile["checkpoints"].append(
            {
                "step_index": np.asarray(simulation.step_count, dtype=np.int64),
                "simulation_time_s": np.asarray(
                    simulation.time_s,
                    dtype=np.float64,
                ),
                "actual_force_n": np.asarray(
                    completed_trial.reaction_force_n,
                    dtype=np.float64,
                ),
                "indentation_m": np.asarray(
                    completed_trial.travel_m - _INITIAL_CLEARANCE_M,
                    dtype=np.float64,
                ),
                "silicone_vertices_m": vertices_m.copy(),
                "contact_indices": records[0].copy(),
                "contact_barycentric": records[1].copy(),
                "contact_normals_W": records[3].copy(),
                "contact_body_positions": records[4].copy(),
                "contact_count": np.asarray(len(records[0]), dtype=np.int64),
                "minimum_det_f": np.asarray(det_f.min(), dtype=np.float64),
                "inversion_count": np.asarray(
                    np.count_nonzero(det_f <= 0.0),
                    dtype=np.int64,
                ),
                "buffer_overflow": np.asarray(overflow, dtype=np.int64),
            }
        )
        if len(profile["checkpoints"]) == len(_FORCE_TARGETS_N):
            profile["loop_end_s"] = perf_counter()
            profile["graph_replays"] = simulation._cuda_graph_replay_count
            _capture_graph_topology(profile, simulation)

    trial = DesignTrial(
        name="sphere_20mm_y+0mm",
        urdf_path=sphere_urdf_path,
        initial_tf=wp.transform(
            wp.vec3(
                0.0,
                0.0,
                fingertip.tip_z_m - _INITIAL_CLEARANCE_M - 10.0e-3,
            ),
            wp.quat_identity(),
        ),
        motion_direction_W=wp.vec3(0.0, 0.0, 1.0),
        approach_speed_m_s=_APPROACH_SPEED_M_S,
        target_force_n=_FORCE_TARGETS_N[-1],
        max_sim_time_s=_MAX_SIM_TIME_S,
        initial_clearance_m=_INITIAL_CLEARANCE_M,
    )
    study = DesignStudy(
        fingertip,
        (trial,),
        fingertip_mesh=fingertip_mesh,
        sim_frequency=_SIM_FREQUENCY_HZ,
        settle_duration_s=_SETTLE_DURATION_S,
        settle_displacement_tolerance_m=None,
        force_tolerance_fraction=_FORCE_TOLERANCE_FRACTION,
        force_targets_n=_FORCE_TARGETS_N,
        force_gain_m_s_n=_FORCE_GAIN_M_S_N,
        element_size_mm=1.0,
        iterations=_VBD_ITERATIONS,
        soft_contact_margin_m=_SOFT_CONTACT_MARGIN_M,
        carrier_contact_stiffness_n_m=_CARRIER_CONTACT_STIFFNESS_N_M,
        contact_stiffness_n_m=_CONTACT_STIFFNESS_N_M,
        contact_damping_n_s_m=_CONTACT_DAMPING_N_S_M,
        use_cuda_graph=use_cuda_graph,
    )

    originals = _install_instrumentation(profile)
    run_start_s = perf_counter()
    try:
        study.run(inspect_trial=collect_checkpoint)
        wp.synchronize()
    finally:
        if profile["detailed_steps_remaining"]:
            profile["detailed_results"] = wp.timing_end(synchronize=True)
            profile["detailed_steps_remaining"] = 0
        run_end_s = perf_counter()
        _restore_instrumentation(originals)

    profile["scenario_wall_s"] = run_end_s - run_start_s
    profile["loop_wall_s"] = profile["loop_end_s"] - profile["loop_start_s"]
    profile["step_count"] = int(trial.step_count)
    profile["simulation_time_s"] = float(trial.simulation_time_s)
    profile["force_trajectory_n"] = np.asarray(
        profile["force_trajectory_n"],
        dtype=np.float64,
    )
    profile["step_event_times_ms"] = [
        float(wp.get_event_elapsed_time(start, end))
        for start, end in profile["step_events"]
    ]
    detailed_events = profile["detailed_events"]
    profile["detailed_window_ms"] = (
        0.0
        if detailed_events is None
        else float(wp.get_event_elapsed_time(*detailed_events))
    )
    return profile


def _difference(
    name: str,
    direct: np.ndarray,
    graph: np.ndarray,
) -> dict[str, object] | None:
    direct = np.asarray(direct)
    graph = np.asarray(graph)
    if direct.shape == graph.shape and np.array_equal(direct, graph):
        return None
    result: dict[str, object] = {
        "name": name,
        "direct_shape": direct.shape,
        "graph_shape": graph.shape,
    }
    common_shape = tuple(min(a, b) for a, b in zip(direct.shape, graph.shape))
    common_slice = tuple(slice(0, size) for size in common_shape)
    direct_common = direct[common_slice]
    graph_common = graph[common_slice]
    unequal = np.not_equal(direct_common, graph_common)
    if not np.any(unequal):
        result.update(
            first_index=f"length mismatch after common shape {common_shape}",
            max_abs_error=0.0,
            max_rel_error=0.0,
        )
        return result
    first_index = tuple(int(value) for value in np.argwhere(unequal)[0])
    result["first_index"] = first_index
    if np.issubdtype(direct.dtype, np.number):
        direct_f = direct_common.astype(np.float64)
        graph_f = graph_common.astype(np.float64)
        absolute = np.abs(direct_f - graph_f)
        denominator = np.maximum(
            np.maximum(np.abs(direct_f), np.abs(graph_f)),
            np.finfo(np.float64).tiny,
        )
        result["max_abs_error"] = float(absolute.max())
        result["max_rel_error"] = float((absolute / denominator).max())
        result["direct_value"] = float(direct_f[first_index])
        result["graph_value"] = float(graph_f[first_index])
    return result


def _compare_profiles(
    direct: dict[str, Any],
    graph: dict[str, Any],
) -> list[dict[str, object]]:
    differences = []
    trajectory_difference = _difference(
        "force_trajectory_n",
        direct["force_trajectory_n"],
        graph["force_trajectory_n"],
    )
    if trajectory_difference is not None:
        first_index = trajectory_difference.get("first_index")
        if isinstance(first_index, tuple) and first_index:
            trajectory_difference["first_timestep"] = first_index[0] + 1
        differences.append(trajectory_difference)

    fields = (
        "step_index",
        "simulation_time_s",
        "actual_force_n",
        "indentation_m",
        "silicone_vertices_m",
        "contact_indices",
        "contact_barycentric",
        "contact_normals_W",
        "contact_body_positions",
        "contact_count",
        "minimum_det_f",
        "inversion_count",
        "buffer_overflow",
    )
    if len(direct["checkpoints"]) != len(graph["checkpoints"]):
        differences.append(
            {
                "name": "checkpoint_count",
                "direct_shape": (len(direct["checkpoints"]),),
                "graph_shape": (len(graph["checkpoints"]),),
                "first_index": "count mismatch",
                "max_abs_error": float("inf"),
                "max_rel_error": float("inf"),
            }
        )
        return differences
    for checkpoint_index, (direct_checkpoint, graph_checkpoint) in enumerate(
        zip(direct["checkpoints"], graph["checkpoints"], strict=True)
    ):
        for field in fields:
            difference = _difference(
                f"checkpoint[{checkpoint_index}].{field}",
                direct_checkpoint[field],
                graph_checkpoint[field],
            )
            if difference is not None:
                difference["checkpoint"] = checkpoint_index
                differences.append(difference)
    return differences


def _activity_summary(profile: dict[str, Any]) -> dict[str, float | int]:
    results = profile["detailed_results"]
    filters = Counter(result.filter for result in results)
    device_filters = (wp.TIMING_KERNEL, wp.TIMING_KERNEL_BUILTIN, wp.TIMING_MEMCPY, wp.TIMING_MEMSET)
    device_activity_ms = sum(
        float(result.elapsed) for result in results if result.filter in device_filters
    )
    return {
        "kernel_launches": filters[wp.TIMING_KERNEL]
        + filters[wp.TIMING_KERNEL_BUILTIN],
        "memcpy_launches": filters[wp.TIMING_MEMCPY],
        "memset_launches": filters[wp.TIMING_MEMSET],
        "graph_launches": filters[wp.TIMING_GRAPH],
        "device_activity_ms": device_activity_ms,
        "window_ms": float(profile["detailed_window_ms"]),
    }


def _write_checkpoint_csv(
    direct: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    path = _OUTPUT_DIRECTORY / "checkpoint_comparison.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "path",
                "target_force_n",
                "step_index",
                "simulation_time_s",
                "actual_force_n",
                "indentation_mm",
                "contact_count",
                "minimum_det_f",
                "inversion_count",
                "buffer_overflow",
            ),
        )
        writer.writeheader()
        for profile in (direct, graph):
            for target_force_n, checkpoint in zip(
                _FORCE_TARGETS_N,
                profile["checkpoints"],
                strict=True,
            ):
                writer.writerow(
                    {
                        "path": profile["label"],
                        "target_force_n": target_force_n,
                        "step_index": int(checkpoint["step_index"]),
                        "simulation_time_s": float(
                            checkpoint["simulation_time_s"]
                        ),
                        "actual_force_n": float(checkpoint["actual_force_n"]),
                        "indentation_mm": 1.0e3
                        * float(checkpoint["indentation_m"]),
                        "contact_count": int(checkpoint["contact_count"]),
                        "minimum_det_f": float(checkpoint["minimum_det_f"]),
                        "inversion_count": int(checkpoint["inversion_count"]),
                        "buffer_overflow": int(checkpoint["buffer_overflow"]),
                    }
                )


def _write_report(
    direct: dict[str, Any],
    direct_repeat: dict[str, Any],
    graph: dict[str, Any],
    differences: list[dict[str, object]],
    repeat_differences: list[dict[str, object]],
) -> None:
    exact = not differences
    direct_repeat_exact = not repeat_differences
    direct_activity = _activity_summary(direct)
    graph_activity = _activity_summary(graph)
    direct_event_ms = mean(direct["step_event_times_ms"])
    graph_event_ms = mean(graph["step_event_times_ms"])
    speedup = direct["loop_wall_s"] / graph["loop_wall_s"]
    direct_busy = (
        direct_activity["device_activity_ms"] / direct_activity["window_ms"]
        if direct_activity["window_ms"] > 0.0
        else 0.0
    )
    estimated_graph_busy = (
        direct_activity["device_activity_ms"] / graph_activity["window_ms"]
        if graph_activity["window_ms"] > 0.0
        else 0.0
    )

    lines = [
        "# Partial CUDA graph equivalence",
        "",
        f"Result: {'PASS' if exact else 'FAIL'}",
        "",
        "## Frozen scenario",
        "",
        "- full five-LED nominal fingertip mesh",
        "- centered 20 mm sphere",
        "- sequential 5/10/15/20 N reference dwell checkpoints",
        "- Newton 100 Hz, SolverVBD 10 iterations, 5 s dwell",
        "- contact endpoints ke=3e4 N/m, kd=0.282280175 N*s/m",
        "- identical CPU force servo and initial state",
        "- graph mode uses one direct warm-up tick, then separate A->B and B->A graphs",
        "",
        "## Exact-equivalence gate",
        "",
        f"- direct vs graph exact result: **{'PASS' if exact else 'FAIL'}**",
        f"- direct vs fresh direct repeat: **{'PASS' if direct_repeat_exact else 'FAIL'}**",
        "",
    ]
    if exact:
        lines.extend(
            [
                "Every requested quantity passed `np.array_equal()`:",
                "",
                "- per-tick reaction-force trajectory",
                "- checkpoint step index and simulation time",
                "- actual force and indentation",
                "- every silicone vertex",
                "- contact primitive vertex IDs and contact count",
                "- barycentric coordinates, normals, and body-local positions",
                "- minimum det(F), inversion count, and contact-buffer overflow",
            ]
        )
    else:
        first = differences[0]
        repeat_first = repeat_differences[0] if repeat_differences else None
        lines.extend(
            [
                f"First differing quantity: `{first['name']}`",
                f"First index: `{first.get('first_index')}`",
                f"First timestep/checkpoint: `{first.get('first_timestep', first.get('checkpoint', 'n/a'))}`",
                f"Maximum absolute error: `{first.get('max_abs_error', 'n/a')}`",
                f"Maximum relative error: `{first.get('max_rel_error', 'n/a')}`",
                "Likely cause under investigation: Newton full-surface contacts use atomically emitted records and atomic wrench accumulation. A fresh direct repeat is also not bitwise reproducible, so graph-specific error must be separated from baseline runtime nondeterminism.",
                "Performance conclusions are withheld because the strict gate did not pass.",
            ]
        )
        if repeat_first is not None:
            lines.extend(
                [
                    "",
                    f"First direct-repeat difference: `{repeat_first['name']}`",
                    f"Direct-repeat first index: `{repeat_first.get('first_index')}`",
                    f"Direct-repeat maximum absolute error: `{repeat_first.get('max_abs_error', 'n/a')}`",
                ]
            )

    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            "| target [N] | step | time [s] | force [N] | indentation [mm] | contacts | min det(F) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target, checkpoint in zip(
        _FORCE_TARGETS_N,
        graph["checkpoints"],
        strict=True,
    ):
        lines.append(
            f"| {target:.0f} | {int(checkpoint['step_index'])} | "
            f"{float(checkpoint['simulation_time_s']):.6f} | "
            f"{float(checkpoint['actual_force_n']):.9f} | "
            f"{1.0e3 * float(checkpoint['indentation_m']):.9f} | "
            f"{int(checkpoint['contact_count'])} | "
            f"{float(checkpoint['minimum_det_f']):.9f} |"
        )

    if exact:
        direct_ops = (
            direct_activity["kernel_launches"]
            + direct_activity["memcpy_launches"]
            + direct_activity["memset_launches"]
        ) / _DETAILED_GPU_WINDOW_STEPS
        lines.extend(
            [
                "",
                "## Performance after equivalence PASS",
                "",
                "| metric | direct | graph |",
                "|---|---:|---:|",
                f"| scenario wall [s] | {direct['scenario_wall_s']:.6f} | {graph['scenario_wall_s']:.6f} |",
                f"| simulation-loop wall [s] | {direct['loop_wall_s']:.6f} | {graph['loop_wall_s']:.6f} |",
                f"| steps/s | {direct['step_count'] / direct['loop_wall_s']:.3f} | {graph['step_count'] / graph['loop_wall_s']:.3f} |",
                f"| sampled CUDA-event step mean [ms] | {direct_event_ms:.6f} | {graph_event_ms:.6f} |",
                f"| sampled CUDA-event step median [ms] | {median(direct['step_event_times_ms']):.6f} | {median(graph['step_event_times_ms']):.6f} |",
                f"| reaction projection/readback wall [s] | {direct['reaction_readback_wall_s']:.6f} | {graph['reaction_readback_wall_s']:.6f} |",
                f"| host device submissions/tick | {direct_ops:.1f} | 1 graph launch |",
                f"| detailed-window kernel launches | {direct_activity['kernel_launches']} | hidden inside {graph_activity['graph_launches']} graph launches |",
                f"| measured direct kernel-busy fraction | {100.0 * direct_busy:.2f}% | n/a inside opaque graph |",
                f"| graph busy fraction estimate from unchanged direct activity | n/a | {100.0 * estimated_graph_busy:.2f}% |",
                "",
                f"Loop speedup: **{speedup:.3f}x**.",
                "",
                "Warp's activity timer reports one opaque `TIMING_GRAPH` record per replay, so it cannot measure individual kernel activity inside a captured graph. The estimate divides the direct path's measured kernel/memcpy/memset activity for the exact same 20 steps by the graph path's CUDA-event window. Exact state/contact equality and the captured graph node inventory verify that collision, VBD, and wrench work was retained.",
                "",
                "### Captured topology",
                "",
            ]
        )
        for direction, counts in graph["graph_nodes"].items():
            lines.append(
                f"- graph_{direction}: {counts['total']} nodes "
                f"({counts['kernel']} kernels, {counts['memcpy']} memcpy, "
                f"{counts['memset']} memset)"
            )

    (_OUTPUT_DIRECTORY / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fingertip = Fingertip(FingertipParameters())
    fingertip_mesh = make_fingertip_5led_mesh(
        fingertip,
        element_size_mm=1.0,
    )
    tet_indices = np.asarray(
        fingertip_mesh.silicone.tet_indices,
        dtype=np.int32,
    ).reshape(-1, 4)
    reference_six_volumes_m3 = _six_tet_volumes(
        np.asarray(fingertip_mesh.silicone.vertices, dtype=np.float32),
        tet_indices,
    )
    resource = files("lumo.assets.objects.urdf").joinpath("sphere_20mm.urdf")
    with as_file(resource) as sphere_urdf_path:
        print("Running direct path...")
        direct = _run_scenario(
            use_cuda_graph=False,
            fingertip=fingertip,
            fingertip_mesh=fingertip_mesh,
            sphere_urdf_path=sphere_urdf_path,
            tet_indices=tet_indices,
            reference_six_volumes_m3=reference_six_volumes_m3,
        )
        print("Running fresh direct repeat...")
        direct_repeat = _run_scenario(
            use_cuda_graph=False,
            fingertip=fingertip,
            fingertip_mesh=fingertip_mesh,
            sphere_urdf_path=sphere_urdf_path,
            tet_indices=tet_indices,
            reference_six_volumes_m3=reference_six_volumes_m3,
        )
        print("Running partial-graph path...")
        graph = _run_scenario(
            use_cuda_graph=True,
            fingertip=fingertip,
            fingertip_mesh=fingertip_mesh,
            sphere_urdf_path=sphere_urdf_path,
            tet_indices=tet_indices,
            reference_six_volumes_m3=reference_six_volumes_m3,
        )

    differences = _compare_profiles(direct, graph)
    repeat_differences = _compare_profiles(direct, direct_repeat)
    np.savez_compressed(
        _OUTPUT_DIRECTORY / "force_trajectories.npz",
        direct=direct["force_trajectory_n"],
        direct_repeat=direct_repeat["force_trajectory_n"],
        graph=graph["force_trajectory_n"],
    )
    _write_checkpoint_csv(direct, graph)
    _write_report(
        direct,
        direct_repeat,
        graph,
        differences,
        repeat_differences,
    )
    print(f"Exact equivalence: {'PASS' if not differences else 'FAIL'}")
    print(
        "Fresh direct repeat: "
        f"{'PASS' if not repeat_differences else 'NOT BITWISE REPRODUCIBLE'}"
    )
    print(f"Direct loop: {direct['loop_wall_s']:.3f} s")
    print(f"Graph loop:  {graph['loop_wall_s']:.3f} s")
    if differences:
        print(f"First difference: {differences[0]}")
        raise RuntimeError("partial CUDA graph exact-equivalence gate failed")
    print(f"Speedup: {direct['loop_wall_s'] / graph['loop_wall_s']:.3f}x")
    print(f"Report: {_OUTPUT_DIRECTORY / 'report.md'}")


if __name__ == "__main__":
    main()
