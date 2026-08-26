"""Prescribed force-threshold indentation for one fingertip."""

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


_PARALLEL_WORLD_COUNT = 4


class IndentationTrial:
    """Definition and current checkpoint result for one indentation scenario."""

    def __init__(
        self,
        *,
        name: str,
        urdf_path: str | Path,
        initial_tf: wp.transform,
        motion_direction_W: wp.vec3,
        approach_speed_m_s: float,
        max_sim_time_s: float,
        initial_clearance_m: float = 0.0,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty string")
        path = Path(urdf_path)
        if path.suffix.lower() != ".urdf":
            raise ValueError("urdf_path must be a .urdf file")
        if not path.is_file():
            raise FileNotFoundError(path)
        require_positive("max_sim_time_s", max_sim_time_s)
        require_positive("approach_speed_m_s", approach_speed_m_s)
        require_nonnegative("initial_clearance_m", initial_clearance_m)

        initial_tf_values = np.asarray(initial_tf, dtype=np.float64)
        if initial_tf_values.shape != (7,) or not np.all(
            np.isfinite(initial_tf_values)
        ):
            raise ValueError("initial_tf must be a finite Warp transform")

        motion_direction = np.asarray(motion_direction_W, dtype=np.float64)
        if motion_direction.shape != (3,) or not np.all(np.isfinite(motion_direction)):
            raise ValueError("motion_direction_W must be a finite 3-vector")
        direction_norm = float(np.linalg.norm(motion_direction))
        require_positive("motion_direction_W norm", direction_norm)
        motion_direction /= direction_norm

        self.name = name.strip()
        self.urdf_path = path
        self.initial_tf = wp.transform(
            wp.vec3(*initial_tf_values[:3]),
            wp.quat(*initial_tf_values[3:]),
        )
        self.motion_direction_W = wp.vec3(*motion_direction)
        self.approach_speed_m_s = float(approach_speed_m_s)
        self.max_sim_time_s = float(max_sim_time_s)
        self.initial_clearance_m = float(initial_clearance_m)

        self.final_tf: wp.transform | None = None
        self.travel_m: float | None = None
        self.step_count = 0
        self.simulation_time_s: float | None = None
        self.reaction_force_n: float | None = None
        self.maximum_particle_speed_m_s: float | None = None
        self.force_change_n: float | None = None
        self.force_threshold_n: float | None = None
        self.force_overshoot_n: float | None = None
        self.reaction_force_rate_n_s: float | None = None
        self.indentation_rate_m_s: float | None = None
        self.wall_runtime_s: float | None = None


class IndentationStudy:
    """Run independent GPU indentation worlds and capture force crossings."""

    def __init__(
        self,
        fingertip: Fingertip,
        trials: Iterable[IndentationTrial],
        *,
        fingertip_mesh: FingertipMesh | None = None,
        sim_frequency: float,
        force_targets_n: Iterable[float],
        element_size_mm: float = 1.0,
        iterations: int = 10,
        soft_contact_margin_m: float = 1.0e-4,
        carrier_contact_stiffness_n_m: float = 1.0e6,
        contact_stiffness_n_m: float | None = None,
        contact_damping_n_s_m: float | None = None,
    ) -> None:
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be a Fingertip")
        if fingertip_mesh is not None:
            if not isinstance(fingertip_mesh, FingertipMesh):
                raise TypeError("fingertip_mesh must be a FingertipMesh")
            if fingertip_mesh.fingertip is not fingertip:
                raise ValueError("fingertip_mesh must belong to the supplied fingertip")
        require_positive("sim_frequency", sim_frequency)
        require_positive("element_size_mm", element_size_mm)
        require_nonnegative("soft_contact_margin_m", soft_contact_margin_m)
        require_positive(
            "carrier_contact_stiffness_n_m", carrier_contact_stiffness_n_m
        )
        if contact_stiffness_n_m is not None:
            require_positive("contact_stiffness_n_m", contact_stiffness_n_m)
        if contact_damping_n_s_m is not None:
            require_nonnegative("contact_damping_n_s_m", contact_damping_n_s_m)
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")

        trial_tuple = tuple(trials)
        if not trial_tuple:
            raise ValueError("trials must contain at least one indentation")
        if any(not isinstance(trial, IndentationTrial) for trial in trial_tuple):
            raise TypeError("trials must contain only IndentationTrial objects")
        if len({id(trial) for trial in trial_tuple}) != len(trial_tuple):
            raise ValueError("trials must not contain the same object twice")
        if any(trial.final_tf is not None for trial in trial_tuple):
            raise ValueError("trials must not have been run already")

        target_tuple = tuple(float(target) for target in force_targets_n)
        if not target_tuple:
            raise ValueError("force_targets_n must not be empty")
        for target in target_tuple:
            require_positive("force target", target)
        if any(
            current <= previous
            for previous, current in zip(target_tuple, target_tuple[1:])
        ):
            raise ValueError("force_targets_n must be strictly increasing")

        self.fingertip = fingertip
        self.fingertip_mesh = fingertip_mesh
        self.trials = trial_tuple
        self.sim_frequency = float(sim_frequency)
        self.force_targets_n = target_tuple
        self.element_size_mm = float(element_size_mm)
        self.iterations = iterations
        self.soft_contact_margin_m = float(soft_contact_margin_m)
        self.carrier_contact_stiffness_n_m = float(
            carrier_contact_stiffness_n_m
        )
        self.contact_stiffness_n_m = (
            None if contact_stiffness_n_m is None else float(contact_stiffness_n_m)
        )
        self.contact_damping_n_s_m = (
            None if contact_damping_n_s_m is None else float(contact_damping_n_s_m)
        )
        self._has_run = False

    @staticmethod
    def _inspect_checkpoint(
        trial: IndentationTrial,
        simulation: LumoSimulation,
        indenter: Indenter,
        checkpoint_index: int,
        initial_translation_W_m: np.ndarray,
        initial_rotation: wp.quat,
        motion_direction_W: np.ndarray,
        inspect_checkpoint: Callable[
            [IndentationTrial, LumoSimulation, Indenter], None
        ]
        | None,
    ) -> None:
        checkpoint = simulation._force_checkpoint(checkpoint_index)
        travel_m = float(checkpoint["travel_m"])
        translation_W_m = initial_translation_W_m + travel_m * motion_direction_W
        trial.final_tf = wp.transform(
            wp.vec3(*translation_W_m),
            initial_rotation,
        )
        trial.travel_m = travel_m
        trial.step_count = int(checkpoint["step_count"])
        trial.simulation_time_s = float(checkpoint["simulation_time_s"])
        trial.reaction_force_n = float(checkpoint["reaction_force_n"])
        trial.force_change_n = float(checkpoint["force_change_n"])
        trial.force_threshold_n = float(checkpoint["force_threshold_n"])
        trial.force_overshoot_n = float(checkpoint["force_overshoot_n"])
        trial.reaction_force_rate_n_s = float(checkpoint["reaction_force_rate_n_s"])
        trial.indentation_rate_m_s = float(checkpoint["indentation_rate_m_s"])
        simulation._select_force_checkpoint(checkpoint_index)
        try:
            trial.maximum_particle_speed_m_s = (
                simulation.maximum_active_particle_speed_m_s()
            )
            if inspect_checkpoint is not None:
                inspect_checkpoint(trial, simulation, indenter)
        finally:
            simulation._restore_live_state()

    def _new_runtime(
        self,
        trial: IndentationTrial,
        *,
        fingertip_model: FingertipNewtonModel | None = None,
        indenter: Indenter | None = None,
    ) -> tuple[LumoSimulation, Indenter]:
        if fingertip_model is None:
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
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
                carrier_contact_stiffness_n_m=self.carrier_contact_stiffness_n_m,
            )
            return simulation, indenter

        if indenter is None:
            raise RuntimeError("a reused fingertip model requires its indenter")
        return (
            LumoSimulation(
                self.fingertip,
                fingertip_model=fingertip_model,
                sim_frequency=self.sim_frequency,
                iterations=self.iterations,
                soft_contact_margin_m=self.soft_contact_margin_m,
                soft_contact_stiffness_n_m=self.contact_stiffness_n_m,
                soft_contact_damping_n_s_m=self.contact_damping_n_s_m,
                element_size_mm=self.element_size_mm,
                carrier_contact_stiffness_n_m=self.carrier_contact_stiffness_n_m,
            ),
            indenter,
        )

    @staticmethod
    def _trial_frame(
        trial: IndentationTrial,
    ) -> tuple[np.ndarray, wp.quat, np.ndarray]:
        initial_tf = np.asarray(trial.initial_tf, dtype=np.float64)
        return (
            initial_tf[:3],
            wp.quat(*initial_tf[3:]),
            np.asarray(trial.motion_direction_W, dtype=np.float64),
        )

    @staticmethod
    def _prepare_runtime(
        simulation: LumoSimulation,
        indenter: Indenter,
        trial: IndentationTrial,
    ) -> None:
        simulation.apply_indenter_pose(indenter, trial.initial_tf)
        simulation.collision_pipeline.collide(simulation.state, simulation.contacts)
        if simulation.soft_contact_count(indenter.body_index):
            raise RuntimeError(
                f"{trial.name} has soft contacts before prescribed motion"
            )

    def _run_parallel_checkpoints(
        self,
        inspect_checkpoint: Callable[
            [IndentationTrial, LumoSimulation, Indenter], None
        ]
        | None,
    ) -> None:
        host_poll_ticks = 2 * max(1, ceil(0.1 * self.sim_frequency))
        grouped_trials: dict[Path, list[IndentationTrial]] = {}
        for trial in self.trials:
            grouped_trials.setdefault(trial.urdf_path.resolve(), []).append(trial)

        for trials in grouped_trials.values():
            fingertip_model: FingertipNewtonModel | None = None
            indenter: Indenter | None = None
            for batch_start in range(0, len(trials), _PARALLEL_WORLD_COUNT):
                batch = trials[batch_start : batch_start + _PARALLEL_WORLD_COUNT]
                worlds: list[dict[str, object]] = []
                batch_start_s = perf_counter()
                for trial in batch:
                    simulation, indenter = self._new_runtime(
                        trial,
                        fingertip_model=fingertip_model,
                        indenter=indenter,
                    )
                    if fingertip_model is None:
                        fingertip_model = simulation.fingertip_model
                    self._prepare_runtime(simulation, indenter, trial)
                    simulation._configure_force_checkpoints(
                        indenter,
                        initial_tf=trial.initial_tf,
                        motion_direction_W=trial.motion_direction_W,
                        approach_speed_m_s=trial.approach_speed_m_s,
                        target_forces_n=self.force_targets_n,
                    )
                    initial_translation, initial_rotation, motion_direction = (
                        self._trial_frame(trial)
                    )
                    worlds.append(
                        {
                            "trial": trial,
                            "simulation": simulation,
                            "indenter": indenter,
                            "stream": wp.Stream(
                                simulation.fingertip_model.model.device
                            ),
                            "processed": 0,
                            "finished": False,
                            "max_steps": int(
                                trial.max_sim_time_s * self.sim_frequency
                            ),
                            "initial_translation": initial_translation,
                            "initial_rotation": initial_rotation,
                            "motion_direction": motion_direction,
                        }
                    )

                while not all(bool(world["finished"]) for world in worlds):
                    active = [world for world in worlds if not world["finished"]]
                    for world in active:
                        simulation = world["simulation"]
                        remaining = int(world["max_steps"]) - simulation.step_count
                        if remaining > 0:
                            simulation._launch_force_checkpoints(
                                min(host_poll_ticks, remaining),
                                stream=world["stream"],
                            )
                    for world in active:
                        simulation = world["simulation"]
                        trial = world["trial"]
                        checkpoint_count, finished, error = (
                            simulation._force_checkpoint_status(
                                stream=world["stream"]
                            )
                        )
                        if error:
                            raise RuntimeError(
                                f"{trial.name} GPU force checkpoints failed with "
                                f"code {error}"
                            )
                        while int(world["processed"]) < checkpoint_count:
                            checkpoint_index = int(world["processed"])
                            if checkpoint_index + 1 == len(self.force_targets_n):
                                trial.wall_runtime_s = perf_counter() - batch_start_s
                            self._inspect_checkpoint(
                                trial,
                                simulation,
                                world["indenter"],
                                checkpoint_index,
                                world["initial_translation"],
                                world["initial_rotation"],
                                world["motion_direction"],
                                inspect_checkpoint,
                            )
                            world["processed"] = checkpoint_index + 1
                        world["finished"] = finished
                        if not finished and simulation.step_count >= int(
                            world["max_steps"]
                        ):
                            missing_index = int(world["processed"])
                            last_force_n = simulation._current_reaction_force_n()
                            raise RuntimeError(
                                f"{trial.name} did not cross "
                                f"{self.force_targets_n[missing_index]:g} N during "
                                f"{trial.max_sim_time_s:g} s of simulation; last "
                                f"force was {last_force_n:.9e} N"
                            )
                batch_wall_s = perf_counter() - batch_start_s
                for world in worlds:
                    world["trial"].wall_runtime_s = batch_wall_s

    def run(
        self,
        *,
        inspect_checkpoint: Callable[
            [IndentationTrial, LumoSimulation, Indenter], None
        ]
        | None = None,
    ) -> None:
        """Run and release each world, inspecting every force crossing."""
        if self._has_run:
            raise RuntimeError("indentation study has already been run")
        if inspect_checkpoint is not None and not callable(inspect_checkpoint):
            raise TypeError("inspect_checkpoint must be callable")
        self._has_run = True
        self._run_parallel_checkpoints(inspect_checkpoint)


__all__ = ["IndentationStudy", "IndentationTrial"]
