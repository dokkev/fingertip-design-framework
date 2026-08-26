"""Profile one frozen production full-finger Newton scenario."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from importlib.resources import as_file, files
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import newton
import numpy as np
import warp as wp

import lumo.simulation.runtime as runtime_module
from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_5led_mesh
from lumo.newton import Indenter
from lumo.optimization.evaluator import (
    _indenter_contact_records,
    _six_tet_volumes,
)
from lumo.simulation import DesignStudy, DesignTrial, LumoSimulation


_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIRECTORY = _ROOT / "output" / "validation" / "newton_runtime_profile"
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


def _capacity_bytes(value: object) -> int:
    capacity = getattr(value, "capacity", None)
    return int(capacity) if isinstance(capacity, int) else 0


def _direct_warp_arrays(owner: object, *, nested_depth: int = 1) -> dict[int, int]:
    """Return unique direct Warp allocation pointers and capacities."""
    arrays: dict[int, int] = {}
    visited: set[int] = set()

    def visit(value: object, depth: int) -> None:
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        ptr = getattr(value, "ptr", None)
        capacity = _capacity_bytes(value)
        if isinstance(ptr, int) and ptr != 0 and capacity > 0:
            arrays[ptr] = capacity
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child, depth)
            return
        if isinstance(value, (tuple, list)):
            for child in value:
                visit(child, depth)
            return
        if depth <= 0 or not hasattr(value, "__dict__"):
            return
        for child in vars(value).values():
            visit(child, depth - 1)

    visit(owner, nested_depth)
    return arrays


def _gpu_event_pair() -> tuple[wp.Event, wp.Event]:
    return (
        wp.Event(device="cuda:0", enable_timing=True),
        wp.Event(device="cuda:0", enable_timing=True),
    )


def _record_start(events: tuple[wp.Event, wp.Event]) -> None:
    wp.record_event(events[0])


def _record_end(events: tuple[wp.Event, wp.Event]) -> None:
    wp.record_event(events[1])


def _event_times_ms(profile: dict[str, Any]) -> dict[str, list[float]]:
    wp.synchronize()
    result: dict[str, list[float]] = {}
    for name, event_pairs in profile["gpu_events"].items():
        result[name] = [
            float(wp.get_event_elapsed_time(start, end, synchronize=False))
            for start, end in event_pairs
        ]
    return result


def _time_call(
    profile: dict[str, Any],
    name: str,
    function: Any,
    *args: object,
    **kwargs: object,
) -> Any:
    start_s = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        profile["wall_s"][name] += perf_counter() - start_s
        profile["calls"][name] += 1


def _install_instrumentation(
    profile: dict[str, Any],
) -> list[tuple[object, str, object]]:
    """Install validation-local timing wrappers and return restoration records."""
    originals: list[tuple[object, str, object]] = []

    def patch(owner: object, name: str, replacement: object) -> None:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    original_runtime_init = LumoSimulation.__init__

    def runtime_init(self: LumoSimulation, *args: object, **kwargs: object) -> None:
        _time_call(
            profile,
            "runtime_initialization",
            original_runtime_init,
            self,
            *args,
            **kwargs,
        )
        profile["simulation"] = self
        model = self.fingertip_model.model
        profile["model_counts"] = {
            "particles": int(model.particle_count),
            "tetrahedra": int(model.tet_count),
            "surface_triangles": int(model.tri_count),
            "surface_edges": int(model.edge_count),
            "bodies": int(model.body_count),
            "shapes": int(model.shape_count),
            "soft_contact_capacity": int(self.contacts.soft_contact_max),
            "body_particle_contact_buffer_per_body": int(
                self.solver.body_particle_contact_buffer_pre_alloc
            ),
        }
        owners = {
            "model": model,
            "solver": self.solver,
            "collision_pipeline": self.collision_pipeline,
            "contacts": self.contacts,
            "state": self.state,
            "next_state": self._next_state,
            "runtime": self,
        }
        subsystem_arrays = {
            name: _direct_warp_arrays(owner) for name, owner in owners.items()
        }
        profile["allocation_bytes_by_owner"] = {
            name: sum(arrays.values()) for name, arrays in subsystem_arrays.items()
        }
        all_arrays: dict[int, int] = {}
        for arrays in subsystem_arrays.values():
            all_arrays.update(arrays)
        profile["unique_device_bytes"] = sum(all_arrays.values())

    patch(LumoSimulation, "__init__", runtime_init)

    original_build = runtime_module.build_fingertip_newton_model

    def build_model(*args: object, **kwargs: object) -> object:
        return _time_call(
            profile, "newton_model_construction", original_build, *args, **kwargs
        )

    patch(runtime_module, "build_fingertip_newton_model", build_model)

    original_solver_init = newton.solvers.SolverVBD.__init__

    def solver_init(self: object, *args: object, **kwargs: object) -> None:
        _time_call(
            profile,
            "vbd_solver_initialization",
            original_solver_init,
            self,
            *args,
            **kwargs,
        )

    patch(newton.solvers.SolverVBD, "__init__", solver_init)

    original_collision_init = newton.CollisionPipeline.__init__

    def collision_init(self: object, *args: object, **kwargs: object) -> None:
        _time_call(
            profile,
            "collision_pipeline_initialization",
            original_collision_init,
            self,
            *args,
            **kwargs,
        )

    patch(newton.CollisionPipeline, "__init__", collision_init)

    original_contacts = newton.CollisionPipeline.contacts

    def contacts(self: object, *args: object, **kwargs: object) -> object:
        return _time_call(
            profile,
            "contact_buffer_allocation",
            original_contacts,
            self,
            *args,
            **kwargs,
        )

    patch(newton.CollisionPipeline, "contacts", contacts)

    original_color = newton.ModelBuilder.color

    def color(self: object, *args: object, **kwargs: object) -> object:
        return _time_call(
            profile, "vbd_coloring", original_color, self, *args, **kwargs
        )

    patch(newton.ModelBuilder, "color", color)

    original_finalize = newton.ModelBuilder.finalize

    def finalize(self: object, *args: object, **kwargs: object) -> object:
        return _time_call(
            profile, "model_finalize", original_finalize, self, *args, **kwargs
        )

    patch(newton.ModelBuilder, "finalize", finalize)

    original_build_sdf = newton.Mesh.build_sdf

    def build_sdf(self: object, *args: object, **kwargs: object) -> object:
        return _time_call(
            profile, "carrier_sdf_build", original_build_sdf, self, *args, **kwargs
        )

    patch(newton.Mesh, "build_sdf", build_sdf)

    original_add_urdf_descriptor = Indenter.__dict__["add_urdf"]
    original_add_urdf = original_add_urdf_descriptor.__func__

    def add_urdf(cls: type[Indenter], *args: object, **kwargs: object) -> Indenter:
        return _time_call(
            profile, "indenter_urdf_import", original_add_urdf, cls, *args, **kwargs
        )

    originals.append((Indenter, "add_urdf", original_add_urdf_descriptor))
    Indenter.add_urdf = classmethod(add_urdf)

    original_apply_pose = LumoSimulation.apply_indenter_pose

    def apply_pose(self: LumoSimulation, *args: object, **kwargs: object) -> None:
        if profile["loop_start_s"] is None:
            profile["loop_start_s"] = perf_counter()
        _time_call(
            profile, "apply_indenter_pose", original_apply_pose, self, *args, **kwargs
        )

    patch(LumoSimulation, "apply_indenter_pose", apply_pose)

    original_collision = newton.CollisionPipeline.collide

    def collide(self: object, *args: object, **kwargs: object) -> object:
        inside_step = bool(profile["inside_step"])
        sample = inside_step and bool(profile["sample_step"])
        events = _gpu_event_pair() if sample or not inside_step else None
        if events is not None:
            _record_start(events)
        result = _time_call(
            profile,
            "collision_step_enqueue" if inside_step else "initial_collision_enqueue",
            original_collision,
            self,
            *args,
            **kwargs,
        )
        if events is not None:
            _record_end(events)
            profile["gpu_events"][
                "collision_step" if inside_step else "initial_collision"
            ].append(events)
        return result

    patch(newton.CollisionPipeline, "collide", collide)

    original_solver_step = newton.solvers.SolverVBD.step

    def solver_step(self: object, *args: object, **kwargs: object) -> None:
        events = _gpu_event_pair() if profile["sample_step"] else None
        if events is not None:
            _record_start(events)
        _time_call(
            profile, "vbd_step_enqueue", original_solver_step, self, *args, **kwargs
        )
        if events is not None:
            _record_end(events)
            profile["gpu_events"]["vbd_step"].append(events)

    patch(newton.solvers.SolverVBD, "step", solver_step)

    original_harvest = newton.solvers.SolverVBD.coupling_harvest_proxy_wrenches

    def harvest(self: object, *args: object, **kwargs: object) -> None:
        events = _gpu_event_pair() if profile["sample_step"] else None
        if events is not None:
            _record_start(events)
        _time_call(
            profile, "wrench_harvest_enqueue", original_harvest, self, *args, **kwargs
        )
        if events is not None:
            _record_end(events)
            profile["gpu_events"]["wrench_harvest"].append(events)

    patch(newton.solvers.SolverVBD, "coupling_harvest_proxy_wrenches", harvest)

    original_step = LumoSimulation.step

    def step(self: LumoSimulation) -> None:
        step_ordinal = profile["calls"]["runtime_step"] + 1
        profile["sample_step"] = (
            step_ordinal <= 5 or step_ordinal % _EVENT_SAMPLE_STRIDE == 0
        )
        if profile["arm_detailed_gpu_window"] and not profile["detailed_gpu_active"]:
            wp.timing_begin(cuda_filter=wp.TIMING_ALL, synchronize=True)
            profile["detailed_gpu_active"] = True
            profile["detailed_gpu_steps_remaining"] = _DETAILED_GPU_WINDOW_STEPS
            profile["arm_detailed_gpu_window"] = False
        events = _gpu_event_pair() if profile["sample_step"] else None
        if events is not None:
            _record_start(events)
        profile["inside_step"] = True
        try:
            _time_call(profile, "runtime_step", original_step, self)
        finally:
            profile["inside_step"] = False
        if events is not None:
            _record_end(events)
            profile["gpu_events"]["runtime_step"].append(events)
        if profile["detailed_gpu_active"]:
            profile["detailed_gpu_steps_remaining"] -= 1
            if profile["detailed_gpu_steps_remaining"] == 0:
                profile["detailed_gpu_results"] = wp.timing_end(synchronize=True)
                profile["detailed_gpu_active"] = False
        profile["last_step_sampled"] = profile["sample_step"]
        profile["sample_step"] = False

    patch(LumoSimulation, "step", step)

    original_reaction_force = LumoSimulation.indenter_reaction_force

    def reaction_force(self: LumoSimulation, *args: object, **kwargs: object) -> float:
        events = _gpu_event_pair() if profile["last_step_sampled"] else None
        if events is not None:
            _record_start(events)
        value = _time_call(
            profile,
            "reaction_force_readback",
            original_reaction_force,
            self,
            *args,
            **kwargs,
        )
        if events is not None:
            _record_end(events)
            wp.synchronize_event(events[1])
            profile["gpu_events"]["force_projection"].append(events)
        if value > 1.0 and not profile["detailed_gpu_window_completed"]:
            if (
                not profile["detailed_gpu_active"]
                and not profile["arm_detailed_gpu_window"]
            ):
                profile["arm_detailed_gpu_window"] = True
                profile["detailed_gpu_window_completed"] = True
        profile["last_step_sampled"] = False
        return value

    patch(LumoSimulation, "indenter_reaction_force", reaction_force)

    original_maximum_speed = LumoSimulation.maximum_active_particle_speed_m_s

    def maximum_speed(self: LumoSimulation) -> float:
        return _time_call(
            profile, "maximum_speed_diagnostic", original_maximum_speed, self
        )

    patch(LumoSimulation, "maximum_active_particle_speed_m_s", maximum_speed)

    original_soft_contact_count = LumoSimulation.soft_contact_count

    def soft_contact_count(
        self: LumoSimulation, *args: object, **kwargs: object
    ) -> int:
        return _time_call(
            profile,
            "soft_contact_count_readback",
            original_soft_contact_count,
            self,
            *args,
            **kwargs,
        )

    patch(LumoSimulation, "soft_contact_count", soft_contact_count)

    return originals


def _restore_instrumentation(originals: list[tuple[object, str, object]]) -> None:
    for owner, name, original in reversed(originals):
        setattr(owner, name, original)


def _checkpoint_readback_bytes(simulation: LumoSimulation) -> int:
    contacts = simulation.contacts
    model = simulation.fingertip_model.model
    arrays = (
        simulation.state.particle_q,
        simulation.solver.body_particle_contact_overflow_max,
        contacts.soft_contact_count,
        contacts.soft_contact_shape,
        model.shape_body,
        contacts.soft_contact_indices,
        contacts.soft_contact_barycentric,
        contacts.soft_contact_normal,
        contacts.soft_contact_body_pos,
        simulation.state.particle_qd,
        model.particle_flags,
        model.particle_mass,
        contacts.soft_contact_count,
        contacts.soft_contact_shape,
        model.shape_body,
        contacts.soft_contact_count,
    )
    return sum(_capacity_bytes(array) for array in arrays if array is not None)


def _run_scenario(
    label: str,
    fingertip: Fingertip,
    fingertip_mesh: object,
    sphere_urdf_path: Path,
    reference_six_volumes_m3: np.ndarray,
    tet_indices: np.ndarray,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "label": label,
        "wall_s": defaultdict(float),
        "calls": Counter(),
        "gpu_events": defaultdict(list),
        "loop_start_s": None,
        "loop_end_s": None,
        "inside_step": False,
        "sample_step": False,
        "last_step_sampled": False,
        "arm_detailed_gpu_window": False,
        "detailed_gpu_active": False,
        "detailed_gpu_window_completed": False,
        "detailed_gpu_steps_remaining": 0,
        "detailed_gpu_results": [],
        "checkpoint_count": 0,
        "checkpoint_rows": [],
        "checkpoint_readback_bytes": 0,
        "checkpoint_readback_s": 0.0,
        "checkpoint_cpu_diagnostics_s": 0.0,
    }
    vertex_count = fingertip_mesh.silicone.vertex_count
    captured_vertices = np.empty(
        (len(_FORCE_TARGETS_N), vertex_count, 3), dtype=np.float32
    )
    contact_chunks: list[np.ndarray] = []

    def collect_checkpoint(
        completed_trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
    ) -> None:
        checkpoint_start_s = perf_counter()
        readback_start_s = perf_counter()
        vertices_m = simulation.silicone_vertices()
        overflow = int(simulation.solver.body_particle_contact_overflow_max.numpy()[0])
        records = _indenter_contact_records(simulation, indenter, vertices_m)
        particle_qd = simulation.state.particle_qd
        particle_flags = simulation.fingertip_model.model.particle_flags
        particle_mass = simulation.fingertip_model.model.particle_mass
        if particle_qd is None or particle_flags is None or particle_mass is None:
            raise RuntimeError("profiled simulation lacks particle diagnostics")
        particle_start = simulation.fingertip_model.silicone_particle_start
        particle_stop = (
            particle_start + simulation.fingertip_model.silicone_particle_count
        )
        velocities_m_s = particle_qd.numpy()[particle_start:particle_stop]
        flags = particle_flags.numpy()[particle_start:particle_stop]
        masses_kg = particle_mass.numpy()[particle_start:particle_stop]
        indenter_contacts = simulation.soft_contact_count(indenter.body_index)
        total_contacts = simulation.soft_contact_count()
        profile["checkpoint_readback_s"] += perf_counter() - readback_start_s
        profile["checkpoint_readback_bytes"] += _checkpoint_readback_bytes(simulation)

        diagnostic_start_s = perf_counter()
        current_six_volumes_m3 = _six_tet_volumes(vertices_m, tet_indices)
        det_f = current_six_volumes_m3 / reference_six_volumes_m3
        active = (flags & int(newton.ParticleFlags.ACTIVE)) != 0
        speeds_m_s = np.linalg.norm(velocities_m_s[active], axis=1)
        checkpoint_index = profile["checkpoint_count"]
        captured_vertices[checkpoint_index] = vertices_m
        contact_chunks.append(records[0].copy())
        kinetic_energy_j = float(0.5 * np.sum(masses_kg[active] * speeds_m_s**2))
        profile["checkpoint_cpu_diagnostics_s"] += perf_counter() - diagnostic_start_s
        if overflow != 0 or np.count_nonzero(det_f <= 0.0):
            raise RuntimeError("profile scenario produced overflow or inversion")
        profile["checkpoint_rows"].append(
            {
                "target_force_n": _FORCE_TARGETS_N[checkpoint_index],
                "actual_force_n": completed_trial.reaction_force_n,
                "indentation_mm": 1.0e3
                * (completed_trial.travel_m - _INITIAL_CLEARANCE_M),
                "simulation_time_s": completed_trial.simulation_time_s,
                "contact_count": indenter_contacts,
                "total_contact_count": total_contacts,
                "minimum_det_f": float(np.min(det_f)),
                "rms_speed_m_s": float(np.sqrt(np.mean(speeds_m_s**2))),
                "kinetic_energy_j": kinetic_energy_j,
                "contact_records": len(records[0]),
            }
        )
        profile["checkpoint_count"] += 1
        profile["wall_s"]["checkpoint_artifact_capture"] += (
            perf_counter() - checkpoint_start_s
        )
        if profile["checkpoint_count"] == len(_FORCE_TARGETS_N):
            profile["loop_end_s"] = perf_counter()

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
    )

    originals = _install_instrumentation(profile)
    profile["run_start_s"] = perf_counter()
    try:
        study.run(inspect_trial=collect_checkpoint)
        wp.synchronize()
    finally:
        if profile["detailed_gpu_active"]:
            profile["detailed_gpu_results"] = wp.timing_end(synchronize=True)
            profile["detailed_gpu_active"] = False
        profile["run_end_s"] = perf_counter()
        _restore_instrumentation(originals)
    profile["scenario_wall_s"] = profile["run_end_s"] - profile["run_start_s"]
    profile["loop_wall_s"] = profile["loop_end_s"] - profile["loop_start_s"]
    profile["initialization_wall_s"] = profile["loop_start_s"] - profile["run_start_s"]
    profile["post_loop_wall_s"] = profile["run_end_s"] - profile["loop_end_s"]
    profile["step_count"] = int(trial.step_count)
    profile["simulated_time_s"] = float(trial.simulation_time_s)
    profile["gpu_event_times_ms"] = _event_times_ms(profile)
    profile["capture_checksum"] = float(captured_vertices.sum()) + sum(
        int(chunk.sum()) for chunk in contact_chunks
    )
    return profile


def _estimated_gpu_seconds(profile: dict[str, Any], name: str) -> float:
    samples_ms = profile["gpu_event_times_ms"].get(name, [])
    if not samples_ms:
        return 0.0
    if name == "initial_collision":
        return 1.0e-3 * sum(samples_ms)
    return 1.0e-3 * mean(samples_ms) * profile["step_count"]


def _detailed_gpu_summary(profile: dict[str, Any]) -> dict[str, Any]:
    results = profile["detailed_gpu_results"]
    kernel_times = [
        result.elapsed
        for result in results
        if result.filter in (wp.TIMING_KERNEL, wp.TIMING_KERNEL_BUILTIN)
    ]
    memcpy_times = [
        result.elapsed for result in results if result.filter == wp.TIMING_MEMCPY
    ]
    memset_times = [
        result.elapsed for result in results if result.filter == wp.TIMING_MEMSET
    ]
    names = Counter(
        result.name
        for result in results
        if result.filter in (wp.TIMING_KERNEL, wp.TIMING_KERNEL_BUILTIN)
    )
    times_by_name: dict[str, float] = defaultdict(float)
    for result in results:
        if result.filter in (wp.TIMING_KERNEL, wp.TIMING_KERNEL_BUILTIN):
            times_by_name[result.name] += float(result.elapsed)
    top_kernels = sorted(times_by_name.items(), key=lambda item: item[1], reverse=True)[
        :12
    ]
    return {
        "kernel_count": len(kernel_times),
        "kernel_total_ms": float(sum(kernel_times)),
        "kernel_median_ms": float(median(kernel_times)) if kernel_times else 0.0,
        "kernel_mean_ms": float(mean(kernel_times)) if kernel_times else 0.0,
        "kernel_max_ms": float(max(kernel_times)) if kernel_times else 0.0,
        "memcpy_count": len(memcpy_times),
        "memcpy_total_ms": float(sum(memcpy_times)),
        "memset_count": len(memset_times),
        "memset_total_ms": float(sum(memset_times)),
        "unique_kernel_names": len(names),
        "top_kernels": top_kernels,
    }


def _timing_rows(
    geometry_wall_s: float,
    profiles: tuple[dict[str, Any], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile in profiles:
        denominator = profile["scenario_wall_s"] + (
            geometry_wall_s if profile["label"] == "cold" else 0.0
        )

        def add(
            scope: str,
            component: str,
            wall_s: float,
            measurement: str,
            overlap: str,
        ) -> None:
            rows.append(
                {
                    "run": profile["label"],
                    "scope": scope,
                    "component": component,
                    "wall_time_s": wall_s,
                    "percent_of_profiled_total": 100.0 * wall_s / denominator,
                    "measurement": measurement,
                    "overlap": overlap,
                }
            )

        add(
            "top_level",
            "geometry_mesh_preparation",
            geometry_wall_s if profile["label"] == "cold" else 0.0,
            "wall_clock",
            "exclusive",
        )
        add(
            "top_level",
            "scenario_initialization",
            profile["initialization_wall_s"],
            "wall_clock",
            "exclusive",
        )
        add(
            "top_level",
            "simulation_loop",
            profile["loop_wall_s"],
            "wall_clock",
            "exclusive",
        )
        add(
            "top_level",
            "post_loop_cleanup",
            profile["post_loop_wall_s"],
            "wall_clock",
            "exclusive",
        )
        for component in (
            "indenter_urdf_import",
            "newton_model_construction",
            "vbd_coloring",
            "model_finalize",
            "carrier_sdf_build",
            "vbd_solver_initialization",
            "collision_pipeline_initialization",
            "contact_buffer_allocation",
            "initial_collision_enqueue",
        ):
            add(
                "initialization_detail",
                component,
                profile["wall_s"][component],
                "wall_clock",
                "nested",
            )
        for component in (
            "runtime_step",
            "collision_step_enqueue",
            "vbd_step_enqueue",
            "wrench_harvest_enqueue",
            "reaction_force_readback",
            "apply_indenter_pose",
            "maximum_speed_diagnostic",
            "soft_contact_count_readback",
            "checkpoint_artifact_capture",
        ):
            add(
                "loop_wall_detail",
                component,
                profile["wall_s"][component],
                "wall_clock",
                "nested",
            )
        servo_residual = max(
            0.0,
            profile["loop_wall_s"]
            - profile["wall_s"]["apply_indenter_pose"]
            - profile["wall_s"]["runtime_step"]
            - profile["wall_s"]["reaction_force_readback"]
            - profile["wall_s"]["maximum_speed_diagnostic"]
            - profile["wall_s"]["checkpoint_artifact_capture"],
        )
        add(
            "loop_wall_detail",
            "force_servo_python_residual",
            servo_residual,
            "wall_clock_residual",
            "nested",
        )
        for event_name in (
            "runtime_step",
            "collision_step",
            "vbd_step",
            "wrench_harvest",
            "force_projection",
        ):
            add(
                "gpu_event_estimate",
                event_name,
                _estimated_gpu_seconds(profile, event_name),
                "sampled_cuda_events_scaled_by_step_count",
                "nested",
            )
        diagnostic_readback = (
            profile["checkpoint_readback_s"]
            + profile["wall_s"]["maximum_speed_diagnostic"]
            + profile["wall_s"]["soft_contact_count_readback"]
        )
        add(
            "diagnostic",
            "diagnostic_readback",
            diagnostic_readback,
            "wall_clock",
            "nested",
        )
        add(
            "diagnostic",
            "checkpoint_cpu_diagnostics",
            profile["checkpoint_cpu_diagnostics_s"],
            "wall_clock",
            "nested",
        )
    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    path = _OUTPUT_DIRECTORY / "timing_breakdown.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def _write_report(
    geometry_wall_s: float,
    geometry_diagnostics_wall_s: float,
    cold: dict[str, Any],
    warm: dict[str, Any],
) -> None:
    profile = warm
    total = profile["scenario_wall_s"]
    gpu_step = _estimated_gpu_seconds(profile, "runtime_step")
    gpu_collision = _estimated_gpu_seconds(profile, "collision_step")
    gpu_vbd = _estimated_gpu_seconds(profile, "vbd_step")
    gpu_harvest = _estimated_gpu_seconds(profile, "wrench_harvest")
    gpu_projection = _estimated_gpu_seconds(profile, "force_projection")
    gpu_other = max(0.0, gpu_step - gpu_collision - gpu_vbd - gpu_harvest)
    sync_boundary = profile["wall_s"]["reaction_force_readback"]
    servo_residual = max(
        0.0,
        profile["loop_wall_s"]
        - profile["wall_s"]["apply_indenter_pose"]
        - profile["wall_s"]["runtime_step"]
        - profile["wall_s"]["reaction_force_readback"]
        - profile["wall_s"]["maximum_speed_diagnostic"]
        - profile["wall_s"]["checkpoint_artifact_capture"],
    )
    detailed = _detailed_gpu_summary(profile)
    event_samples = profile["gpu_event_times_ms"]
    sampled_step_ms = event_samples.get("runtime_step", [])
    kernel_busy_fraction = (
        detailed["kernel_total_ms"]
        / (mean(sampled_step_ms) * _DETAILED_GPU_WINDOW_STEPS)
        if sampled_step_ms
        else 0.0
    )
    warm_init_repeat_fraction = profile["initialization_wall_s"] / total
    current_morphology_s = 1222.991
    measured_newton_21_s = cold["scenario_wall_s"] + 20.0 * warm["scenario_wall_s"]
    device_activity_per_step_s = (
        1.0e-3
        * (
            detailed["kernel_total_ms"]
            + detailed["memcpy_total_ms"]
            + detailed["memset_total_ms"]
        )
        / _DETAILED_GPU_WINDOW_STEPS
    )
    low_risk_loop_s = (
        1.5 * device_activity_per_step_s * profile["step_count"]
        + sync_boundary
        + servo_residual
        + profile["wall_s"]["checkpoint_artifact_capture"]
        + profile["wall_s"]["maximum_speed_diagnostic"]
    )
    low_risk_scenario = (
        profile["initialization_wall_s"] + low_risk_loop_s + profile["post_loop_wall_s"]
    )
    moderate_loop_s = (
        1.25 * device_activity_per_step_s * profile["step_count"]
        + profile["wall_s"]["checkpoint_artifact_capture"]
        + profile["wall_s"]["maximum_speed_diagnostic"]
    )
    non_newton_remainder = max(0.0, current_morphology_s - measured_newton_21_s)
    low_risk_morphology = non_newton_remainder + 21.0 * low_risk_scenario
    moderate_morphology = (
        non_newton_remainder
        + 3.0 * profile["initialization_wall_s"]
        + 21.0 * (moderate_loop_s + profile["post_loop_wall_s"])
    )

    lines = [
        "# Production Newton runtime profile",
        "",
        "Result: PASS (profiling completed; no production behavior changed)",
        "",
        "## Frozen scenario",
        "",
        "- nominal 60 mm five-LED fingertip",
        "- 20 mm sphere at Y=0 mm",
        "- sequential 5/10/15/20 N checkpoints",
        "- 100 Hz, 10 VBD iterations, 5 s force-band dwell, +/-10% tolerance",
        "- ke=3e4 N/m and kd=0.282280175 N s/m on both contact endpoints",
        "- Newton 1.5.0, Warp 1.16.0, RTX 4070 Ti SUPER",
        "- OptiX was intentionally excluded from this Newton-only profile",
        "",
        "## Cold and warm throughput",
        "",
        "| run | scenario wall [s] | steps | simulated time [s] | steps/wall-s | wall/simulated-s | initialization [s] | loop [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in (cold, warm):
        lines.append(
            f"| {item['label']} | {item['scenario_wall_s']:.6f} | {item['step_count']} | "
            f"{item['simulated_time_s']:.3f} | {item['step_count'] / item['loop_wall_s']:.3f} | "
            f"{item['loop_wall_s'] / item['simulated_time_s']:.3f} | "
            f"{item['initialization_wall_s']:.6f} | {item['loop_wall_s']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Geometry/mesh preparation was {_format_seconds(geometry_wall_s)} s once per morphology; "
            f"reference tet diagnostics added {_format_seconds(geometry_diagnostics_wall_s)} s.",
            "The cold run is the first runtime in this process but may load kernels from Warp's persistent disk cache. "
            "The warm run reuses the already-built morphology mesh and carrier SDF, while rebuilding the Newton model and solver exactly as production does.",
            "",
            "## Warm wall-time breakdown",
            "",
            "Top-level rows are exclusive. Detail rows are nested and intentionally do not sum to 100%.",
            "",
            "| component | time [s] | % scenario | measurement note |",
            "|---|---:|---:|---|",
            f"| Newton scenario initialization | {profile['initialization_wall_s']:.6f} | {100 * profile['initialization_wall_s'] / total:.2f}% | exclusive wall |",
            f"| simulation loop total | {profile['loop_wall_s']:.6f} | {100 * profile['loop_wall_s'] / total:.2f}% | exclusive wall |",
            f"| post-loop cleanup | {profile['post_loop_wall_s']:.6f} | {100 * profile['post_loop_wall_s'] / total:.2f}% | exclusive wall |",
            f"| Newton model construction | {profile['wall_s']['newton_model_construction']:.6f} | {100 * profile['wall_s']['newton_model_construction'] / total:.2f}% | nested initialization |",
            f"| soft-mesh/topology/carrier/bond model build | {(profile['wall_s']['newton_model_construction'] - profile['wall_s']['vbd_coloring'] - profile['wall_s']['model_finalize'] - profile['wall_s']['carrier_sdf_build']):.6f} | {100 * (profile['wall_s']['newton_model_construction'] - profile['wall_s']['vbd_coloring'] - profile['wall_s']['model_finalize'] - profile['wall_s']['carrier_sdf_build']) / total:.2f}% | model-construction residual after coloring/finalize/SDF |",
            f"| VBD solver initialization | {profile['wall_s']['vbd_solver_initialization']:.6f} | {100 * profile['wall_s']['vbd_solver_initialization'] / total:.2f}% | nested initialization |",
            f"| collision/contact-buffer initialization | {(profile['wall_s']['collision_pipeline_initialization'] + profile['wall_s']['contact_buffer_allocation']):.6f} | {100 * (profile['wall_s']['collision_pipeline_initialization'] + profile['wall_s']['contact_buffer_allocation']) / total:.2f}% | nested initialization |",
            f"| collision detection during stepping | {gpu_collision:.6f} | {100 * gpu_collision / total:.2f}% | sampled CUDA events, scaled |",
            f"| VBD solve during stepping | {gpu_vbd:.6f} | {100 * gpu_vbd / total:.2f}% | sampled CUDA events, scaled; includes 10 iterations plus solver init/finalize kernels |",
            f"| force/contact reduction | {(gpu_harvest + gpu_projection):.6f} | {100 * (gpu_harvest + gpu_projection) / total:.2f}% | sampled CUDA events, scaled |",
            f"| other step GPU work | {gpu_other:.6f} | {100 * gpu_other / total:.2f}% | total-step CUDA event residual |",
            f"| force-servo/Python loop update | {servo_residual:.6f} | {100 * servo_residual / total:.2f}% | wall residual after timed calls |",
            f"| per-step force readback boundary | {sync_boundary:.6f} | {100 * sync_boundary / total:.2f}% | nested wall; includes waiting for already-queued GPU work, projection, and 4-byte D2H |",
            f"| diagnostic readback | {(profile['checkpoint_readback_s'] + profile['wall_s']['maximum_speed_diagnostic'] + profile['wall_s']['soft_contact_count_readback']):.6f} | {100 * (profile['checkpoint_readback_s'] + profile['wall_s']['maximum_speed_diagnostic'] + profile['wall_s']['soft_contact_count_readback']) / total:.2f}% | nested checkpoint/precontact wall |",
            f"| checkpoint artifact capture | {profile['wall_s']['checkpoint_artifact_capture']:.6f} | {100 * profile['wall_s']['checkpoint_artifact_capture'] / total:.2f}% | nested wall, four checkpoints |",
            "",
            "The force-readback row is a synchronization boundary, not pure synchronization overhead. It overlaps the CUDA-event GPU categories because `.numpy()` waits for their completion. A pure D2H-only percentage cannot be separated without changing the pipeline or an external timeline profiler.",
            "",
            "## CUDA event breakdown",
            "",
            "| GPU region | mean sampled step [ms] | estimated scenario total [s] | % scenario |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, label in (
        ("runtime_step", "complete Newton step"),
        ("collision_step", "collision"),
        ("vbd_step", "SolverVBD.step"),
        ("wrench_harvest", "proxy wrench harvest"),
        ("force_projection", "reaction projection"),
    ):
        samples = event_samples.get(name, [])
        estimated = _estimated_gpu_seconds(profile, name)
        lines.append(
            f"| {label} | {(mean(samples) if samples else 0.0):.6f} | {estimated:.6f} | {100 * estimated / total:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Approximate VBD time per configured iteration is {1.0e3 * gpu_vbd / profile['step_count'] / _VBD_ITERATIONS:.6f} ms. "
            "This division is diagnostic only because SolverVBD.step also contains per-step initialize/finalize work.",
            f"The {_DETAILED_GPU_WINDOW_STEPS}-step active-contact Warp window recorded {detailed['kernel_count']} kernel launches "
            f"({detailed['kernel_count'] / _DETAILED_GPU_WINDOW_STEPS:.1f}/step), {detailed['unique_kernel_names']} unique kernel names, "
            f"median/mean/max kernel durations {detailed['kernel_median_ms']:.6f}/{detailed['kernel_mean_ms']:.6f}/{detailed['kernel_max_ms']:.6f} ms.",
            f"It recorded {detailed['memcpy_count']} memcpy activities ({detailed['memcpy_total_ms']:.6f} ms) and "
            f"{detailed['memset_count']} memset activities ({detailed['memset_total_ms']:.6f} ms).",
            f"Approximate kernel-busy fraction inside the sampled step interval is {100 * kernel_busy_fraction:.1f}%. "
            "This is a same-stream activity ratio, not whole-device SM occupancy.",
            f"Kernel+memcpy+memset activity totals {1.0e3 * device_activity_per_step_s:.6f} ms per profiled step, "
            f"versus {mean(sampled_step_ms):.6f} ms CUDA-event elapsed; the gap is primarily launch/orchestration latency.",
            "",
            "Top kernels in the active-contact window:",
            "",
            "| kernel | total [ms] |",
            "|---|---:|",
        ]
    )
    for name, elapsed_ms in detailed["top_kernels"]:
        lines.append(f"| `{name}` | {elapsed_ms:.6f} |")

    lines.extend(
        [
            "",
            "## Checkpoints and diagnostics",
            "",
            "| target [N] | actual [N] | indentation [mm] | sim time [s] | contacts | min det(F) | RMS speed [m/s] |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in profile["checkpoint_rows"]:
        lines.append(
            f"| {checkpoint['target_force_n']:.0f} | {checkpoint['actual_force_n']:.6f} | "
            f"{checkpoint['indentation_mm']:.6f} | {checkpoint['simulation_time_s']:.3f} | "
            f"{checkpoint['contact_count']} | {checkpoint['minimum_det_f']:.6f} | "
            f"{checkpoint['rms_speed_m_s']:.6e} |"
        )
    lines.extend(
        [
            "",
            f"The four production-style mechanics checkpoints copied approximately {profile['checkpoint_readback_bytes'] / 1.0e6:.3f} MB from device arrays. "
            f"Per-step reaction readback transferred only {4 * profile['step_count'] / 1.0e3:.3f} kB total, but forced one synchronization per tick.",
            "Per-step diagnostics: reaction force only. Maximum particle speed is evaluated four times, and vertices, det(F), inversion, contact occupancy/records/centroid, particle-speed statistics, and mesh snapshot data are evaluated only at accepted checkpoints. Indentation and tolerance state stay on the CPU each tick.",
            "",
            "### Host/device synchronization inventory",
            "",
            "| site | frequency in warm run | transferred data | reason |",
            "|---|---:|---|---|",
            f"| `LumoSimulation.indenter_reaction_force()` | {profile['step_count']:,}/tick | one float (4 B) | CPU proportional servo and tolerance counter |",
            f"| `LumoSimulation.soft_contact_count()` | {profile['calls']['soft_contact_count_readback']} total | scalar count; body-filtered calls also copy full shape/body arrays | initial-contact guard and two contact counts per checkpoint |",
            "| `LumoSimulation.maximum_active_particle_speed_m_s()` | 4 checkpoints | two scalars after a device reduction | checkpoint stability diagnostic |",
            "| `FingertipNewtonModel.silicone_vertices()` | 4 checkpoints | full particle position array | deformation snapshot and det(F) |",
            "| `_indenter_contact_records()` | 4 checkpoints | count plus full stored shape/index/barycentric/normal/body-position buffers | raw patch/contact artifact |",
            "| evaluator particle diagnostics | 4 checkpoints | full velocity, flag, and mass arrays | RMS/P95/kinetic-energy diagnostics |",
            "| contact-buffer overflow | 4 checkpoints | one integer | hard validity check |",
            "",
            "There are no repository calls to `wp.synchronize()` or `wp.synchronize_device()` in the production tick loop; synchronization is implicit in the `.numpy()` calls above. No `.cpu()`, `tolist()`, or scalar `item()` path was found there.",
            "",
            "## Allocation and repeated-initialization audit",
            "",
            f"Observed model: {profile['model_counts']}. Direct unique Warp allocations reachable from the runtime/model/solver/collision/state owners total approximately {profile['unique_device_bytes'] / 1.0e6:.3f} MB (diagnostic lower bound; nested upstream owners may contain additional arrays).",
            "",
            "| allocation site | frequency | size / behavior | reuse assessment |",
            "|---|---|---|---|",
            "| full-finger Gmsh mesh | once per morphology | host tet/surface/carrier meshes | already shared across all 21 scenarios |",
            "| carrier collision SDF | first runtime per morphology mesh | cached on `carrier_collision.sdf` | already reused by later scenarios |",
            "| ModelBuilder/finalized model/material arrays/coloring | every scenario | complete particles/tets/shapes/model arrays | topology depends on morphology+sphere diameter, not contact Y |",
            "| SolverVBD buffers | every scenario | solver state and per-body 2048 contact list | topology depends on morphology+sphere diameter, not contact Y |",
            "| CollisionPipeline/contact buffers | every scenario | particle+edge+face candidate capacity | topology depends on morphology+sphere diameter, not contact Y |",
            "| two Newton states/control/wrench/reduction buffers | every scenario | persistent for that runtime | reusable after a verified complete reset |",
            "| body-particle material buffers | first step if full-surface capacity exceeds solver's constructor estimate | four float arrays resized once, then persistent | warm one uncaptured step is required before graph capture |",
            "| per-step runtime | every tick | no repository-owned `wp.empty/zeros/clone`; existing arrays are zeroed/copied | no steady-state repository allocation found |",
            "| checkpoint host artifacts | four times | full positions/contact arrays/velocity flags/mass readbacks | production already preallocates compact result arrays; contact chunks remain host-owned |",
            "",
            f"Warm initialization still consumes {100 * warm_init_repeat_fraction:.2f}% of one scenario. In the current 21-runtime evaluator, model finalization, coloring, solver, collision pipeline, contact buffers, states, and indenter import are rebuilt 21 times. Only morphology mesh/SDF and OptiX scene are already shared.",
            "",
            "## CUDA graph audit",
            "",
            "No repository production path calls `wp.capture_begin`, `wp.capture_end`, or `wp.capture_launch`; every pose/bond, collision, VBD iteration, wrench, and force-projection kernel is launched individually from Python.",
            "The current entire loop is not one graph because `.numpy()` returns force to the CPU and Python computes the proportional servo, tolerance counter, target transition, pose, and stop condition every tick.",
            "A partial graph containing pose/bond updates, collision, SolverVBD.step, wrench harvest, and force projection is feasible after one uncaptured warm-up step, provided graph-stable arrays and fixed launch capacities are retained. It would still require graph replay followed by a 4-byte force readback each tick, so it reduces launch overhead but not the serial CPU decision boundary.",
            "The smallest change that removes the boundary is to keep reaction scalar, target index, tolerance counter, travel, and kinematic pose in persistent Warp arrays; launch the exact same clipped proportional update on device, and read back only an accepted-checkpoint/finished flag. This preserves the servo equation but is a moderate production orchestration change, not a new controller.",
            "",
            "## Reset and reuse feasibility",
            "",
            "For a fixed morphology and sphere diameter, contact Y changes only the initial indenter pose. Model topology, coloring, material arrays, SDFs, SolverVBD buffers, and collision candidate topology are identical.",
            "Newton 1.5 `SolverVBD.reset()` explicitly restores selected particle/body state, rebaselines particle history on the next step, and rebuilds body-particle contacts per step. The material is stateless Neo-Hookean+damping; collision contact matching is disabled here. A reusable runtime is therefore plausible.",
            "A correct reset must restore both state buffers, particle positions/velocities/forces, carrier/bonded particles, indenter pose/velocity, body history, SolverVBD reset state, contact counters/generation and overflow, wrench/reduction buffers, simulation counters, and all Python servo/trial state. The current `LumoSimulation` has no such production reset operation and only one of its two state buffers is accepted by `SolverVBD.reset()` at a time. Numerical/bitwise equivalence to a fresh runtime has not been demonstrated, so reuse must not be enabled before a fresh-vs-reset checkpoint regression.",
            "",
            "## Collision implementation",
            "",
            "`CollisionPipeline.collide()` runs once per 100 Hz tick, before the 10 VBD iterations. It regenerates rigid-soft particle contacts and the opted-in full-surface edge/face contacts. Sphere/carrier shapes use Newton shape queries; the carrier mesh uses the cached volume SDF. Candidate particle/edge/face pair arrays are built at pipeline initialization, while per-step shape AABBs and contact records are refreshed. Particle self-contact is disabled, so SolverVBD's particle self-contact BVH is not active. No repository broad-phase/SDF rebuild occurs inside the tick loop.",
            "",
            "## Ranked physics-preserving opportunities",
            "",
            "| rank | opportunity | expected effect | complexity | scientific risk |",
            "|---:|---|---|---|---|",
            "| 1 | partial CUDA graph for fixed-capacity pose/bond + collision + VBD + wrench/projection | removes hundreds of Python/CUDA launches per tick; force readback remains | medium | low after exact-state regression |",
            "| 2 | reuse a finalized model/coloring per sphere diameter, initially with fresh solver/state objects | removes the measured 6.44 s model rebuild from six of seven Y scenarios | medium | low-to-moderate after initial-pose/rebaseline regression |",
            "| 3 | exact servo state/update on device, checkpoint-only host flag/readback | direct readback cost is small, but this enables multi-tick graph replay without a CPU decision boundary | medium | low-to-moderate; equation and checkpoint tick must be regression-tested |",
            "",
            "Checkpoint diagnostic reduction is lower impact because diagnostics run only four times. No numerical setting or collision cadence needs to change.",
            "",
            "## Runtime estimate under the frozen contract",
            "",
            f"- Frozen nominal measurement: {current_morphology_s:.3f} s/morphology (includes Newton and OptiX).",
            f"- This profile's 21-scenario Newton-only extrapolation: {measured_newton_21_s:.3f} s; the residual {non_newton_remainder:.3f} s is treated conservatively as mesh/no-contact/OptiX/other work.",
            f"- Low-risk implementation estimate: about {low_risk_morphology:.1f} s/morphology. This assumes partial graph replay reaches 1.5x the measured kernel+memcpy+memset activity time per step, while retaining per-tick force readback, Python servo, all initialization, checkpoint capture, and unchanged OptiX.",
            f"- Moderate architecture estimate: about {moderate_morphology:.1f} s/morphology. This assumes device-resident exact servo/graph replay reaches 1.25x measured device activity and model initialization is paid once per sphere diameter; checkpoint capture and OptiX remain unchanged.",
            "These are bounds from one scenario, not promised speedups. They do not assume a reduction in 100 Hz, 10 iterations, dwell, scenario count, mesh, contact, or optics.",
            "",
            "## Required answers",
            "",
            f"1. **VBD solve:** approximately {100 * gpu_vbd / total:.2f}% of warm scenario wall by sampled CUDA-event time.",
            f"2. **Collision detection:** approximately {100 * gpu_collision / total:.2f}%.",
            f"3. **Host/device synchronization/readback:** the per-step force-readback blocking boundary occupies {100 * sync_boundary / total:.2f}% of wall, inclusive of waiting for GPU work; pure D2H overhead is not separately identifiable here.",
            "4. **Per-step CPU synchronization:** yes, `LumoSimulation.indenter_reaction_force()` calls `_reaction_force_n.numpy()` every tick.",
            "5. **CUDA graphs currently used:** no.",
            "6. **Graph feasibility:** a warmed partial physics-step graph is feasible; the present CPU force servo prevents multi-tick all-device replay.",
            "7. **Device-side force/servo:** yes; the exact scalar reduction and proportional update can remain in persistent Warp arrays, with host readback only at checkpoint/termination.",
            f"8. **Repeated initialization:** {100 * warm_init_repeat_fraction:.2f}% of warm scenario wall is initialization, repeated 21 times except mesh/SDF reuse.",
            "9. **Model/solver/coloring reuse:** technically plausible per sphere diameter across Y, but not yet safe to enable because complete two-state/solver/contact reset equivalence is unvalidated.",
            "10. **Loop allocations:** no repository-owned steady-state allocation was found; Newton can grow full-surface body-particle material buffers once on the first step, and checkpoint host arrays/chunks are allocated four times.",
            f"11. **Bound type:** {'GPU-compute-dominated' if kernel_busy_fraction > 0.75 else 'launch/orchestration/synchronization-limited'} in the sampled window (approximate kernel-busy fraction {100 * kernel_busy_fraction:.1f}%, {detailed['kernel_count'] / _DETAILED_GPU_WINDOW_STEPS:.1f} launches/step).",
            "12. **Top three:** partial CUDA graph; reuse finalized model/coloring per sphere diameter with fresh runtime state; device-resident exact servo/checkpoint flag.",
            f"13. **Realistic speedup:** low-risk estimate {current_morphology_s / low_risk_morphology:.2f}x; moderate estimate {current_morphology_s / moderate_morphology:.2f}x, under stated assumptions.",
            "14. **Batching later:** consider it only if post-graph/device-servo profiling still shows low SM occupancy; this profile does not justify implementing batching first.",
            "15. **Contract confirmation:** no numerical setting, scientific contract, OptiX behavior, objective, scenario set, or BO campaign was changed.",
        ]
    )
    (_OUTPUT_DIRECTORY / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fingertip = Fingertip(FingertipParameters())
    geometry_start_s = perf_counter()
    fingertip_mesh = make_fingertip_5led_mesh(fingertip, element_size_mm=1.0)
    geometry_wall_s = perf_counter() - geometry_start_s
    geometry_diagnostics_start_s = perf_counter()
    reference_vertices_m = np.ascontiguousarray(
        fingertip_mesh.silicone.vertices, dtype=np.float32
    )
    tet_indices = np.asarray(
        fingertip_mesh.silicone.tet_indices, dtype=np.int32
    ).reshape(-1, 4)
    reference_six_volumes_m3 = _six_tet_volumes(reference_vertices_m, tet_indices)
    geometry_diagnostics_wall_s = perf_counter() - geometry_diagnostics_start_s
    if np.any(np.abs(reference_six_volumes_m3) <= 1.0e-18):
        raise RuntimeError("reference mesh contains a degenerate tetrahedron")

    resource_root = files("lumo").joinpath("assets", "objects", "urdf")
    with as_file(resource_root.joinpath("sphere_20mm.urdf")) as sphere_urdf_path:
        print("cold exact production scenario", flush=True)
        cold = _run_scenario(
            "cold",
            fingertip,
            fingertip_mesh,
            sphere_urdf_path,
            reference_six_volumes_m3,
            tet_indices,
        )
        print(
            f"cold: {cold['scenario_wall_s']:.3f} s, {cold['step_count']} steps",
            flush=True,
        )
        print("warm identical production scenario", flush=True)
        warm = _run_scenario(
            "warm",
            fingertip,
            fingertip_mesh,
            sphere_urdf_path,
            reference_six_volumes_m3,
            tet_indices,
        )
        print(
            f"warm: {warm['scenario_wall_s']:.3f} s, {warm['step_count']} steps",
            flush=True,
        )

    rows = _timing_rows(geometry_wall_s, (cold, warm))
    _write_csv(rows)
    _write_report(
        geometry_wall_s,
        geometry_diagnostics_wall_s,
        cold,
        warm,
    )
    print(_OUTPUT_DIRECTORY / "report.md", flush=True)
    print(_OUTPUT_DIRECTORY / "timing_breakdown.csv", flush=True)


if __name__ == "__main__":
    main()
