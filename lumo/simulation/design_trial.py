"""Design trials that evaluate one fingertip design."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import ceil
from pathlib import Path
from time import perf_counter

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.mesh import FingertipMesh
from lumo.newton import FingertipNewtonModel, Indenter
from lumo.simulation.runtime import LumoSimulation
from lumo.util.scalar_validation import require_nonnegative, require_positive


_DEFAULT_FORCE_GAIN_M_S_N = 1.25e-3
REFERENCE_DWELL_LOADING = "reference_dwell"
QUASISTATIC_RAMP_LOADING = "quasistatic_ramp"
FIRST_CROSSING_LOADING = "first_crossing"


class DesignTrial:
    """Definition and lightweight result of one indentation trial."""

    def __init__(
        self,
        *,
        name: str,
        urdf_path: str | Path,
        initial_tf: wp.transform,
        motion_direction_W: wp.vec3,
        approach_speed_m_s: float,
        target_force_n: float,
        max_sim_time_s: float,
        initial_clearance_m: float | None = 0.0,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty string")
        path = Path(urdf_path)
        if path.suffix.lower() != ".urdf":
            raise ValueError("urdf_path must be a .urdf file")
        if not path.is_file():
            raise FileNotFoundError(path)
        require_positive("target_force_n", target_force_n)
        require_positive("max_sim_time_s", max_sim_time_s)
        if initial_clearance_m is not None:
            require_nonnegative("initial_clearance_m", initial_clearance_m)

        initial_tf_values = np.asarray(initial_tf, dtype=np.float64)
        if initial_tf_values.shape != (7,) or not np.all(
            np.isfinite(initial_tf_values)
        ):
            raise ValueError("initial_tf must be a finite Warp transform")

        motion_direction = np.asarray(
            motion_direction_W,
            dtype=np.float64,
        )
        if motion_direction.shape != (3,) or not np.all(np.isfinite(motion_direction)):
            raise ValueError("motion_direction_W must be a finite 3-vector")
        motion_direction_norm = float(np.linalg.norm(motion_direction))
        require_positive("motion_direction_W norm", motion_direction_norm)
        require_positive("approach_speed_m_s", approach_speed_m_s)
        motion_direction /= motion_direction_norm

        self.name = name.strip()
        self.urdf_path = path
        self.initial_tf = wp.transform(
            wp.vec3(*initial_tf_values[:3]),
            wp.quat(*initial_tf_values[3:]),
        )
        self.motion_direction_W = wp.vec3(*motion_direction)
        self.approach_speed_m_s = float(approach_speed_m_s)
        self.target_force_n = float(target_force_n)
        self.max_sim_time_s = float(max_sim_time_s)
        self.initial_clearance_m = (
            None if initial_clearance_m is None else float(initial_clearance_m)
        )

        self.final_tf: wp.transform | None = None
        self.travel_m: float | None = None
        self.step_count = 0
        self.simulation_time_s: float | None = None
        self.reaction_force_n: float | None = None
        self.maximum_particle_speed_m_s: float | None = None
        self.force_change_n: float | None = None
        self.force_reference_n: float | None = None
        self.reaction_force_rate_n_s: float | None = None
        self.indentation_rate_m_s: float | None = None
        self.servo_error_n: float | None = None
        self.settle_window_force_drift_n: float | None = None
        self.settle_window_indentation_drift_m: float | None = None
        self.wall_runtime_s: float | None = None


class DesignStudy:
    """Evaluate one fingertip design with independent trials."""

    def __init__(
        self,
        fingertip: Fingertip,
        trials: Iterable[DesignTrial],
        *,
        fingertip_mesh: FingertipMesh | None = None,
        sim_frequency: float,
        settle_duration_s: float,
        settle_displacement_tolerance_m: float | None = None,
        force_tolerance_n: float | None = None,
        force_tolerance_fraction: float | None = None,
        force_targets_n: Iterable[float] | None = None,
        force_gain_m_s_n: float = _DEFAULT_FORCE_GAIN_M_S_N,
        element_size_mm: float = 1.0,
        iterations: int = 10,
        soft_contact_margin_m: float = 1.0e-4,
        carrier_contact_stiffness_n_m: float = 1.0e6,
        contact_stiffness_n_m: float | None = None,
        contact_damping_n_s_m: float | None = None,
        use_cuda_graph: bool = False,
        reuse_finalized_models: bool = False,
        reuse_runtimes: bool = False,
        parallel_world_count: int = 1,
    ) -> None:
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be a Fingertip")
        if fingertip_mesh is not None:
            if not isinstance(fingertip_mesh, FingertipMesh):
                raise TypeError("fingertip_mesh must be a FingertipMesh")
            if fingertip_mesh.fingertip is not fingertip:
                raise ValueError("fingertip_mesh must belong to the supplied fingertip")
        require_positive("sim_frequency", sim_frequency)
        require_nonnegative("settle_duration_s", settle_duration_s)
        if settle_displacement_tolerance_m is not None:
            require_nonnegative(
                "settle_displacement_tolerance_m",
                settle_displacement_tolerance_m,
            )
        if (force_tolerance_n is None) == (force_tolerance_fraction is None):
            raise ValueError(
                "provide exactly one of force_tolerance_n or force_tolerance_fraction"
            )
        if force_tolerance_n is not None:
            require_positive("force_tolerance_n", force_tolerance_n)
        if force_tolerance_fraction is not None:
            require_positive(
                "force_tolerance_fraction",
                force_tolerance_fraction,
            )
            if force_tolerance_fraction >= 1.0:
                raise ValueError("force_tolerance_fraction must be smaller than 1")
        require_positive("force_gain_m_s_n", force_gain_m_s_n)
        require_positive("element_size_mm", element_size_mm)
        require_nonnegative(
            "soft_contact_margin_m",
            soft_contact_margin_m,
        )
        require_positive(
            "carrier_contact_stiffness_n_m",
            carrier_contact_stiffness_n_m,
        )
        if contact_stiffness_n_m is not None:
            require_positive(
                "contact_stiffness_n_m",
                contact_stiffness_n_m,
            )
        if contact_damping_n_s_m is not None:
            require_nonnegative(
                "contact_damping_n_s_m",
                contact_damping_n_s_m,
            )
        if not isinstance(use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be a bool")
        if not isinstance(reuse_finalized_models, bool):
            raise TypeError("reuse_finalized_models must be a bool")
        if not isinstance(reuse_runtimes, bool):
            raise TypeError("reuse_runtimes must be a bool")
        if reuse_runtimes and not use_cuda_graph:
            raise ValueError("runtime reuse requires use_cuda_graph=True")
        if (
            isinstance(parallel_world_count, bool)
            or not isinstance(parallel_world_count, int)
            or parallel_world_count < 1
        ):
            raise ValueError("parallel_world_count must be a positive integer")
        if parallel_world_count > 1 and not use_cuda_graph:
            raise ValueError("parallel worlds require use_cuda_graph=True")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")

        trial_tuple = tuple(trials)
        if not trial_tuple:
            raise ValueError("trials must contain at least one indentation")
        if any(not isinstance(trial, DesignTrial) for trial in trial_tuple):
            raise TypeError("trials must contain only DesignTrial objects")
        if len({id(trial) for trial in trial_tuple}) != len(trial_tuple):
            raise ValueError("trials must not contain the same object twice")
        if any(trial.final_tf is not None for trial in trial_tuple):
            raise ValueError("trials must not have been run already")

        target_tuple = (
            None
            if force_targets_n is None
            else tuple(float(target) for target in force_targets_n)
        )
        if target_tuple is not None:
            if not target_tuple:
                raise ValueError("force_targets_n must not be empty")
            for target in target_tuple:
                require_positive("force target", target)
            if any(
                current <= previous
                for previous, current in zip(
                    target_tuple,
                    target_tuple[1:],
                )
            ):
                raise ValueError("force_targets_n must be strictly increasing")
            if any(
                not np.isclose(
                    trial.target_force_n,
                    target_tuple[-1],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for trial in trial_tuple
            ):
                raise ValueError(
                    "each trial target_force_n must equal the final force target"
                )

        self.fingertip = fingertip
        self.fingertip_mesh = fingertip_mesh
        self.trials = trial_tuple
        self.sim_frequency = float(sim_frequency)
        self.force_tolerance_n = (
            None if force_tolerance_n is None else float(force_tolerance_n)
        )
        self.force_tolerance_fraction = (
            None
            if force_tolerance_fraction is None
            else float(force_tolerance_fraction)
        )
        self.force_targets_n = target_tuple
        self.settle_duration_s = float(settle_duration_s)
        self.settle_displacement_tolerance_m = (
            None
            if settle_displacement_tolerance_m is None
            else float(settle_displacement_tolerance_m)
        )
        self.force_gain_m_s_n = float(force_gain_m_s_n)
        self.element_size_mm = float(element_size_mm)
        self.iterations = iterations
        self.soft_contact_margin_m = float(soft_contact_margin_m)
        self.carrier_contact_stiffness_n_m = float(carrier_contact_stiffness_n_m)
        self.contact_stiffness_n_m = (
            None if contact_stiffness_n_m is None else float(contact_stiffness_n_m)
        )
        self.contact_damping_n_s_m = (
            None if contact_damping_n_s_m is None else float(contact_damping_n_s_m)
        )
        self.use_cuda_graph = use_cuda_graph
        self.reuse_finalized_models = reuse_finalized_models or reuse_runtimes
        self.reuse_runtimes = reuse_runtimes
        self.parallel_world_count = parallel_world_count
        self._has_run = False

    @staticmethod
    def _inspect_graph_checkpoint(
        trial: DesignTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
        checkpoint_index: int,
        initial_translation_W_m: np.ndarray,
        initial_rotation: wp.quat,
        motion_direction_W: np.ndarray,
        inspect_trial: Callable[[DesignTrial, LumoSimulation, Indenter], None]
        | None,
    ) -> None:
        checkpoint = simulation.force_servo_checkpoint(checkpoint_index)
        travel_m = float(checkpoint["travel_m"])
        translation_W_m = (
            initial_translation_W_m + travel_m * motion_direction_W
        )
        trial.final_tf = wp.transform(
            wp.vec3(*translation_W_m),
            initial_rotation,
        )
        trial.travel_m = travel_m
        trial.step_count = int(checkpoint["step_count"])
        trial.simulation_time_s = float(checkpoint["simulation_time_s"])
        trial.reaction_force_n = float(checkpoint["reaction_force_n"])
        trial.force_change_n = float(checkpoint["force_change_n"])
        trial.force_reference_n = float(checkpoint["force_reference_n"])
        trial.reaction_force_rate_n_s = float(
            checkpoint["reaction_force_rate_n_s"]
        )
        trial.indentation_rate_m_s = float(checkpoint["indentation_rate_m_s"])
        trial.servo_error_n = float(checkpoint["servo_error_n"])
        trial.settle_window_force_drift_n = float(
            checkpoint["settle_window_force_drift_n"]
        )
        trial.settle_window_indentation_drift_m = float(
            checkpoint["settle_window_indentation_drift_m"]
        )
        simulation.select_force_servo_checkpoint(checkpoint_index)
        try:
            trial.maximum_particle_speed_m_s = (
                simulation.maximum_active_particle_speed_m_s()
            )
            if inspect_trial is not None:
                inspect_trial(trial, simulation, indenter)
        finally:
            simulation.restore_force_servo_live_state()

    def _run_parallel_graph(
        self,
        loading_mode: str,
        inspect_trial: Callable[[DesignTrial, LumoSimulation, Indenter], None]
        | None,
    ) -> None:
        """Run independent same-sphere histories on separate CUDA streams."""
        force_targets_n = self.force_targets_n
        if force_targets_n is None:
            raise ValueError("parallel worlds require an explicit force schedule")
        force_tolerances_n = tuple(
            self.force_tolerance_n
            if self.force_tolerance_n is not None
            else target_force_n * self.force_tolerance_fraction
            for target_force_n in force_targets_n
        )
        first_crossing = loading_mode == FIRST_CROSSING_LOADING
        settle_ticks = (
            1
            if first_crossing
            else max(1, ceil(self.settle_duration_s * self.sim_frequency))
        )
        host_poll_ticks = max(2, 2 * ceil(0.1 * self.sim_frequency))
        grouped_trials: dict[Path, list[DesignTrial]] = {}
        for trial in self.trials:
            grouped_trials.setdefault(trial.urdf_path.resolve(), []).append(trial)

        for trials in grouped_trials.values():
            fingertip_model: FingertipNewtonModel | None = None
            indenter: Indenter | None = None
            for batch_start in range(0, len(trials), self.parallel_world_count):
                batch = trials[batch_start : batch_start + self.parallel_world_count]
                worlds: list[dict[str, object]] = []
                batch_start_s = perf_counter()
                for trial in batch:
                    initial_tf = np.asarray(trial.initial_tf, dtype=np.float64)
                    initial_translation_W_m = initial_tf[:3]
                    initial_rotation = wp.quat(*initial_tf[3:])
                    motion_direction_W = np.asarray(
                        trial.motion_direction_W,
                        dtype=np.float64,
                    )
                    if fingertip_model is None:
                        builder = newton.ModelBuilder(
                            gravity=wp.vec3(0.0, 0.0, 0.0)
                        )
                        indenter = Indenter.add_urdf(
                            builder,
                            trial.urdf_path,
                            tf=trial.initial_tf,
                            contact_stiffness_n_m=self.contact_stiffness_n_m,
                            contact_damping_n_s_m=self.contact_damping_n_s_m,
                        )
                        simulation = LumoSimulation(
                            self.fingertip,
                            builder=builder,
                            fingertip_mesh=self.fingertip_mesh,
                            sim_frequency=self.sim_frequency,
                            iterations=self.iterations,
                            soft_contact_margin_m=self.soft_contact_margin_m,
                            soft_contact_stiffness_n_m=(
                                self.contact_stiffness_n_m
                            ),
                            soft_contact_damping_n_s_m=(
                                self.contact_damping_n_s_m
                            ),
                            element_size_mm=self.element_size_mm,
                            carrier_contact_stiffness_n_m=(
                                self.carrier_contact_stiffness_n_m
                            ),
                            use_cuda_graph=True,
                        )
                        fingertip_model = simulation.fingertip_model
                    else:
                        if indenter is None:
                            raise RuntimeError("parallel indenter is unavailable")
                        simulation = LumoSimulation(
                            self.fingertip,
                            fingertip_model=fingertip_model,
                            sim_frequency=self.sim_frequency,
                            iterations=self.iterations,
                            soft_contact_margin_m=self.soft_contact_margin_m,
                            soft_contact_stiffness_n_m=(
                                self.contact_stiffness_n_m
                            ),
                            soft_contact_damping_n_s_m=(
                                self.contact_damping_n_s_m
                            ),
                            element_size_mm=self.element_size_mm,
                            carrier_contact_stiffness_n_m=(
                                self.carrier_contact_stiffness_n_m
                            ),
                            use_cuda_graph=True,
                        )
                    if indenter is None:
                        raise RuntimeError("parallel indenter was not constructed")
                    simulation.apply_indenter_pose(indenter, trial.initial_tf)
                    simulation.collision_pipeline.collide(
                        simulation.state,
                        simulation.contacts,
                    )
                    if simulation.soft_contact_count(indenter.body_index):
                        raise RuntimeError(
                            f"{trial.name} has soft contacts before prescribed motion"
                        )
                    if trial.initial_clearance_m is None:
                        raise ValueError(
                            "GPU-resident force servo requires initial_clearance_m"
                        )
                    simulation.configure_force_servo(
                        indenter,
                        initial_tf=trial.initial_tf,
                        motion_direction_W=trial.motion_direction_W,
                        approach_speed_m_s=trial.approach_speed_m_s,
                        force_gain_m_s_n=self.force_gain_m_s_n,
                        target_forces_n=force_targets_n,
                        force_tolerances_n=force_tolerances_n,
                        settle_ticks=settle_ticks,
                        displacement_tolerance_m=(
                            self.settle_displacement_tolerance_m
                        ),
                        capture_on_first_crossing=first_crossing,
                    )
                    worlds.append(
                        {
                            "trial": trial,
                            "simulation": simulation,
                            "stream": wp.Stream(simulation.fingertip_model.model.device),
                            "processed": 0,
                            "finished": False,
                            "max_steps": int(
                                trial.max_sim_time_s * self.sim_frequency
                            ),
                            "initial_translation": initial_translation_W_m,
                            "initial_rotation": initial_rotation,
                            "motion_direction": motion_direction_W,
                        }
                    )

                while not all(bool(world["finished"]) for world in worlds):
                    active = [world for world in worlds if not world["finished"]]
                    for world in active:
                        simulation = world["simulation"]
                        remaining = int(world["max_steps"]) - simulation.step_count
                        if remaining <= 0:
                            continue
                        simulation.launch_force_servo(
                            min(host_poll_ticks, remaining),
                            stream=world["stream"],
                        )
                    for world in active:
                        simulation = world["simulation"]
                        trial = world["trial"]
                        checkpoint_count, finished, servo_error = (
                            simulation.force_servo_status(stream=world["stream"])
                        )
                        if servo_error:
                            raise RuntimeError(
                                f"{trial.name} GPU force servo failed with code "
                                f"{servo_error}"
                            )
                        while int(world["processed"]) < checkpoint_count:
                            checkpoint_index = int(world["processed"])
                            if checkpoint_index + 1 == len(force_targets_n):
                                trial.wall_runtime_s = (
                                    perf_counter() - batch_start_s
                                )
                            self._inspect_graph_checkpoint(
                                trial,
                                simulation,
                                indenter,
                                checkpoint_index,
                                world["initial_translation"],
                                world["initial_rotation"],
                                world["motion_direction"],
                                inspect_trial,
                            )
                            world["processed"] = checkpoint_index + 1
                        world["finished"] = finished
                        if not finished and simulation.step_count >= int(
                            world["max_steps"]
                        ):
                            last_force_n = simulation.force_servo_current_force_n()
                            missing_index = int(world["processed"])
                            if first_crossing:
                                raise RuntimeError(
                                    f"{trial.name} did not cross "
                                    f"{force_targets_n[missing_index]:g} N during "
                                    f"{trial.max_sim_time_s:g} s of simulation; "
                                    f"last force was {last_force_n:.9e} N"
                                )
                            raise RuntimeError(
                                f"{trial.name} did not keep "
                                f"{force_targets_n[missing_index]:g} N within "
                                f"{force_tolerances_n[missing_index]:g} N for "
                                f"{self.settle_duration_s:g} s during "
                                f"{trial.max_sim_time_s:g} s of simulation; last "
                                f"force was {last_force_n:.9e} N"
                            )
                batch_wall_s = perf_counter() - batch_start_s
                for world in worlds:
                    world["trial"].wall_runtime_s = batch_wall_s

    def run(
        self,
        *,
        loading_mode: str = REFERENCE_DWELL_LOADING,
        force_ramp_rate_n_s: float | None = None,
        inspect_trial: Callable[
            [DesignTrial, LumoSimulation, Indenter],
            None,
        ]
        | None = None,
    ) -> None:
        """Run and release trials, inspecting every accepted force target."""
        if self._has_run:
            raise RuntimeError("design study has already been run")
        if inspect_trial is not None and not callable(inspect_trial):
            raise TypeError("inspect_trial must be callable")
        if loading_mode not in {
            REFERENCE_DWELL_LOADING,
            QUASISTATIC_RAMP_LOADING,
            FIRST_CROSSING_LOADING,
        }:
            raise ValueError(
                "loading_mode must be 'reference_dwell', 'quasistatic_ramp', "
                "or 'first_crossing'"
            )
        if loading_mode == REFERENCE_DWELL_LOADING:
            require_positive("settle_duration_s", self.settle_duration_s)
            if force_ramp_rate_n_s is not None:
                raise ValueError(
                    "force_ramp_rate_n_s is only valid for quasistatic_ramp"
                )
        elif loading_mode == QUASISTATIC_RAMP_LOADING:
            if force_ramp_rate_n_s is None:
                raise ValueError(
                    "quasistatic_ramp requires force_ramp_rate_n_s"
                )
            require_positive("force_ramp_rate_n_s", force_ramp_rate_n_s)
        else:
            if self.settle_duration_s != 0.0:
                raise ValueError("first_crossing requires settle_duration_s=0")
            if force_ramp_rate_n_s is not None:
                raise ValueError(
                    "force_ramp_rate_n_s is only valid for quasistatic_ramp"
                )
        self._has_run = True
        if self.parallel_world_count > 1:
            if loading_mode == QUASISTATIC_RAMP_LOADING:
                raise ValueError(
                    "parallel worlds do not support quasistatic_ramp"
                )
            self._run_parallel_graph(loading_mode, inspect_trial)
            return

        finalized_models: dict[
            Path,
            tuple[FingertipNewtonModel, Indenter],
        ] = {}
        runtime_cache: dict[Path, tuple[LumoSimulation, Indenter]] = {}
        for trial in self.trials:
            initial_tf = np.asarray(trial.initial_tf, dtype=np.float64)
            initial_translation_W_m = initial_tf[:3]
            initial_rotation = wp.quat(*initial_tf[3:])
            motion_direction_W = np.asarray(
                trial.motion_direction_W,
                dtype=np.float64,
            )
            max_step_count = int(trial.max_sim_time_s * self.sim_frequency)
            if max_step_count < 1:
                raise ValueError(
                    "max_sim_time_s must include at least one simulation tick"
                )
            first_crossing = loading_mode == FIRST_CROSSING_LOADING
            settle_ticks = (
                1
                if first_crossing
                else max(1, ceil(self.settle_duration_s * self.sim_frequency))
            )

            model_key = trial.urdf_path.resolve()
            cached_runtime = runtime_cache.get(model_key)
            runtime_was_reset = cached_runtime is not None
            if cached_runtime is not None:
                simulation, indenter = cached_runtime
                builder = None
                simulation.reset_force_servo(
                    indenter,
                    initial_tf=trial.initial_tf,
                )
            else:
                cached = finalized_models.get(model_key)
                if cached is None:
                    builder = newton.ModelBuilder(
                        gravity=wp.vec3(0.0, 0.0, 0.0)
                    )
                    indenter = Indenter.add_urdf(
                        builder,
                        trial.urdf_path,
                        tf=trial.initial_tf,
                        contact_stiffness_n_m=self.contact_stiffness_n_m,
                        contact_damping_n_s_m=self.contact_damping_n_s_m,
                    )
                    simulation = LumoSimulation(
                        self.fingertip,
                        builder=builder,
                        fingertip_mesh=self.fingertip_mesh,
                        sim_frequency=self.sim_frequency,
                        iterations=self.iterations,
                        soft_contact_margin_m=self.soft_contact_margin_m,
                        soft_contact_stiffness_n_m=self.contact_stiffness_n_m,
                        soft_contact_damping_n_s_m=self.contact_damping_n_s_m,
                        element_size_mm=self.element_size_mm,
                        carrier_contact_stiffness_n_m=(
                            self.carrier_contact_stiffness_n_m
                        ),
                        use_cuda_graph=self.use_cuda_graph,
                    )
                    if self.reuse_finalized_models:
                        finalized_models[model_key] = (
                            simulation.fingertip_model,
                            indenter,
                        )
                else:
                    fingertip_model, indenter = cached
                    builder = None
                    simulation = LumoSimulation(
                        self.fingertip,
                        fingertip_model=fingertip_model,
                        sim_frequency=self.sim_frequency,
                        iterations=self.iterations,
                        soft_contact_margin_m=self.soft_contact_margin_m,
                        soft_contact_stiffness_n_m=self.contact_stiffness_n_m,
                        soft_contact_damping_n_s_m=self.contact_damping_n_s_m,
                        element_size_mm=self.element_size_mm,
                        carrier_contact_stiffness_n_m=(
                            self.carrier_contact_stiffness_n_m
                        ),
                        use_cuda_graph=self.use_cuda_graph,
                    )
                if self.reuse_runtimes:
                    runtime_cache[model_key] = (simulation, indenter)

            simulation.apply_indenter_pose(indenter, trial.initial_tf)
            simulation.collision_pipeline.collide(
                simulation.state,
                simulation.contacts,
            )

            if simulation.soft_contact_count(indenter.body_index):
                raise RuntimeError(
                    f"{trial.name} has soft contacts before prescribed motion"
                )

            force_targets_n = self.force_targets_n or (trial.target_force_n,)
            force_tolerances_n = tuple(
                self.force_tolerance_n
                if self.force_tolerance_n is not None
                else target_force_n * self.force_tolerance_fraction
                for target_force_n in force_targets_n
            )
            if self.use_cuda_graph:
                if loading_mode == QUASISTATIC_RAMP_LOADING:
                    raise ValueError(
                        "GPU-resident checkpoints do not support quasistatic_ramp"
                    )
                if trial.initial_clearance_m is None:
                    raise ValueError(
                        "GPU-resident force servo requires initial_clearance_m"
                    )
                if not runtime_was_reset:
                    simulation.configure_force_servo(
                        indenter,
                        initial_tf=trial.initial_tf,
                        motion_direction_W=trial.motion_direction_W,
                        approach_speed_m_s=trial.approach_speed_m_s,
                        force_gain_m_s_n=self.force_gain_m_s_n,
                        target_forces_n=force_targets_n,
                        force_tolerances_n=force_tolerances_n,
                        settle_ticks=settle_ticks,
                        displacement_tolerance_m=(
                            self.settle_displacement_tolerance_m
                        ),
                        capture_on_first_crossing=first_crossing,
                    )
                processed_checkpoint_count = 0
                finished = False
                # Poll every 0.2 physical seconds. Accepted ticks are copied to
                # device snapshots, so this cadence does not approximate them.
                host_poll_ticks = max(2, 2 * ceil(0.1 * self.sim_frequency))
                while simulation.step_count < max_step_count and not finished:
                    tick_count = min(
                        host_poll_ticks,
                        max_step_count - simulation.step_count,
                    )
                    checkpoint_count, finished, servo_error = (
                        simulation.advance_force_servo(tick_count)
                    )
                    if servo_error:
                        raise RuntimeError(
                            f"{trial.name} GPU force servo failed with code "
                            f"{servo_error}"
                        )
                    while processed_checkpoint_count < checkpoint_count:
                        checkpoint_index = processed_checkpoint_count
                        self._inspect_graph_checkpoint(
                            trial,
                            simulation,
                            indenter,
                            checkpoint_index,
                            initial_translation_W_m,
                            initial_rotation,
                            motion_direction_W,
                            inspect_trial,
                        )
                        processed_checkpoint_count += 1

                if not finished:
                    last_force_n = simulation.force_servo_current_force_n()
                    missing_target_n = force_targets_n[
                        processed_checkpoint_count
                    ]
                    if first_crossing:
                        raise RuntimeError(
                            f"{trial.name} did not cross {missing_target_n:g} N "
                            f"during {trial.max_sim_time_s:g} s of simulation; "
                            f"last force was {last_force_n:.9e} N"
                        )
                    missing_tolerance_n = force_tolerances_n[
                        processed_checkpoint_count
                    ]
                    raise RuntimeError(
                        f"{trial.name} did not keep {missing_target_n:g} N "
                        f"within {missing_tolerance_n:g} N for "
                        f"{self.settle_duration_s:g} s during "
                        f"{trial.max_sim_time_s:g} s of simulation; last "
                        f"force was {last_force_n:.9e} N"
                    )
                del simulation, builder
                continue

            travel_m = 0.0
            reaction_force_n = 0.0
            force_change_n = float("inf")
            previous_force_n = 0.0
            in_tolerance_ticks = 0
            pose = trial.initial_tf
            target_index = 0
            target_force_n = force_targets_n[target_index]
            force_tolerance_n = force_tolerances_n[target_index]

            while simulation.step_count < max_step_count:
                force_reference_n = target_force_n
                if loading_mode == QUASISTATIC_RAMP_LOADING:
                    force_reference_n = min(
                        force_targets_n[-1],
                        force_ramp_rate_n_s * simulation.time_s,
                    )
                if first_crossing:
                    velocity_m_s = trial.approach_speed_m_s
                else:
                    force_error_n = force_reference_n - reaction_force_n
                    minimum_velocity_m_s = (
                        0.0
                        if loading_mode == QUASISTATIC_RAMP_LOADING
                        else -trial.approach_speed_m_s
                    )
                    velocity_m_s = max(
                        minimum_velocity_m_s,
                        min(
                            trial.approach_speed_m_s,
                            self.force_gain_m_s_n * force_error_n,
                        ),
                    )
                commanded_displacement_m = velocity_m_s * simulation.time_step_s
                travel_m += commanded_displacement_m
                translation_W_m = (
                    initial_translation_W_m + travel_m * motion_direction_W
                )
                pose = wp.transform(
                    wp.vec3(*translation_W_m),
                    initial_rotation,
                )
                simulation.apply_indenter_pose(indenter, pose)

                simulation.step()
                reaction_force_n = simulation.indenter_reaction_force(
                    indenter,
                    motion_direction_W=trial.motion_direction_W,
                )
                if trial.initial_clearance_m is None and reaction_force_n > 0.0:
                    trial.initial_clearance_m = travel_m
                signed_force_change_n = reaction_force_n - previous_force_n
                force_change_n = abs(signed_force_change_n)
                reaction_force_rate_n_s = (
                    signed_force_change_n / simulation.time_step_s
                )
                previous_force_n = reaction_force_n

                if first_crossing:
                    if reaction_force_n < target_force_n:
                        continue
                elif loading_mode == QUASISTATIC_RAMP_LOADING:
                    if reaction_force_n < target_force_n:
                        continue
                    if abs(reaction_force_n - target_force_n) > force_tolerance_n:
                        raise RuntimeError(
                            f"{trial.name} crossed {target_force_n:g} N at "
                            f"{reaction_force_n:.9e} N, outside the "
                            f"+/-{force_tolerance_n:g} N capture tolerance"
                        )
                else:
                    force_is_settled = (
                        abs(reaction_force_n - target_force_n) <= force_tolerance_n
                    )
                    displacement_is_settled = (
                        self.settle_displacement_tolerance_m is None
                        or abs(commanded_displacement_m)
                        <= self.settle_displacement_tolerance_m
                    )
                    if force_is_settled and displacement_is_settled:
                        in_tolerance_ticks += 1
                    else:
                        in_tolerance_ticks = 0

                    if in_tolerance_ticks < settle_ticks:
                        continue

                if trial.initial_clearance_m is None:
                    raise RuntimeError(f"{trial.name} reached force without contact")
                trial.final_tf = pose
                trial.travel_m = travel_m
                trial.step_count = simulation.step_count
                trial.simulation_time_s = simulation.time_s
                trial.reaction_force_n = reaction_force_n
                trial.maximum_particle_speed_m_s = (
                    simulation.maximum_active_particle_speed_m_s()
                )
                trial.force_change_n = force_change_n
                trial.force_reference_n = force_reference_n
                trial.reaction_force_rate_n_s = reaction_force_rate_n_s
                trial.indentation_rate_m_s = velocity_m_s
                trial.servo_error_n = force_reference_n - reaction_force_n
                if inspect_trial is not None:
                    inspect_trial(trial, simulation, indenter)
                target_index += 1
                if target_index == len(force_targets_n):
                    break
                target_force_n = force_targets_n[target_index]
                force_tolerance_n = force_tolerances_n[target_index]
                in_tolerance_ticks = 0
            else:
                if first_crossing:
                    raise RuntimeError(
                        f"{trial.name} did not cross {target_force_n:g} N "
                        f"during {trial.max_sim_time_s:g} s of simulation; "
                        f"last force was {reaction_force_n:.9e} N"
                    )
                if loading_mode == QUASISTATIC_RAMP_LOADING:
                    raise RuntimeError(
                        f"{trial.name} did not cross {target_force_n:g} N "
                        f"within tolerance during {trial.max_sim_time_s:g} s "
                        f"of continuous loading; last force was "
                        f"{reaction_force_n:.9e} N"
                    )
                displacement_condition = ""
                if self.settle_displacement_tolerance_m is not None:
                    displacement_condition = (
                        " while commanding no more than "
                        f"{self.settle_displacement_tolerance_m:g} m/tick"
                    )
                raise RuntimeError(
                    f"{trial.name} did not keep {target_force_n:g} N "
                    f"within {force_tolerance_n:g} N for "
                    f"{self.settle_duration_s:g} s{displacement_condition} "
                    f"during {trial.max_sim_time_s:g} s of simulation; last "
                    f"force was {reaction_force_n:.9e} N"
                )

            del simulation, builder


__all__ = [
    "DesignStudy",
    "DesignTrial",
    "FIRST_CROSSING_LOADING",
    "QUASISTATIC_RAMP_LOADING",
    "REFERENCE_DWELL_LOADING",
]
