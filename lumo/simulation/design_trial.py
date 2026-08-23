"""Design trials that evaluate one fingertip design."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import ceil
from pathlib import Path

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.newton import Indenter
from lumo.simulation.runtime import LumoSimulation
from lumo.util.scalar_validation import require_nonnegative, require_positive


class DesignTrial:
    """Definition and lightweight result of one indentation trial."""

    def __init__(
        self,
        *,
        name: str,
        urdf_path: str | Path,
        initial_tf: wp.transform,
        translation_step_W_m: wp.vec3,
        target_force_n: float,
        max_sim_time_s: float,
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

        initial_tf_values = np.asarray(initial_tf, dtype=np.float64)
        if initial_tf_values.shape != (7,) or not np.all(
            np.isfinite(initial_tf_values)
        ):
            raise ValueError("initial_tf must be a finite Warp transform")

        translation_step = np.asarray(
            translation_step_W_m,
            dtype=np.float64,
        )
        if translation_step.shape != (3,) or not np.all(
            np.isfinite(translation_step)
        ):
            raise ValueError(
                "translation_step_W_m must be a finite 3-vector"
            )
        translation_step_norm_m = float(
            np.linalg.norm(translation_step)
        )
        require_positive(
            "translation_step_W_m norm",
            translation_step_norm_m,
        )

        self.name = name.strip()
        self.urdf_path = path
        self.initial_tf = wp.transform(
            wp.vec3(*initial_tf_values[:3]),
            wp.quat(*initial_tf_values[3:]),
        )
        self.translation_step_W_m = wp.vec3(*translation_step)
        self.target_force_n = float(target_force_n)
        self.max_sim_time_s = float(max_sim_time_s)

        self.final_tf: wp.transform | None = None
        self.travel_m: float | None = None
        self.step_count = 0
        self.simulation_time_s: float | None = None
        self.search_iteration_count = 0
        self.reaction_force_n: float | None = None
        self.maximum_particle_speed_m_s: float | None = None
        self.force_change_n: float | None = None


class DesignStudy:
    """Evaluate one fingertip design with independent trials."""

    def __init__(
        self,
        fingertip: Fingertip,
        trials: Iterable[DesignTrial],
        *,
        sim_frequency: float,
        force_tolerance_n: float,
        force_duration_s: float,
        max_search_iterations: int,
        element_size_mm: float = 1.0,
        iterations: int = 10,
        soft_contact_margin_m: float = 1.0e-4,
        carrier_contact_stiffness_n_m: float = 1.0e6,
    ) -> None:
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be a Fingertip")
        require_positive("sim_frequency", sim_frequency)
        require_positive("force_tolerance_n", force_tolerance_n)
        require_positive("force_duration_s", force_duration_s)
        require_positive("element_size_mm", element_size_mm)
        require_nonnegative(
            "soft_contact_margin_m",
            soft_contact_margin_m,
        )
        require_positive(
            "carrier_contact_stiffness_n_m",
            carrier_contact_stiffness_n_m,
        )
        for name, value in (
            ("max_search_iterations", max_search_iterations),
            ("iterations", iterations),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        trial_tuple = tuple(trials)
        if not trial_tuple:
            raise ValueError("trials must contain at least one indentation")
        if any(
            not isinstance(trial, DesignTrial)
            for trial in trial_tuple
        ):
            raise TypeError("trials must contain only DesignTrial objects")
        if len({id(trial) for trial in trial_tuple}) != len(trial_tuple):
            raise ValueError("trials must not contain the same object twice")
        if any(trial.final_tf is not None for trial in trial_tuple):
            raise ValueError("trials must not have been run already")

        self.fingertip = fingertip
        self.trials = trial_tuple
        self.sim_frequency = float(sim_frequency)
        self.force_tolerance_n = float(force_tolerance_n)
        self.force_duration_s = float(force_duration_s)
        self.max_search_iterations = max_search_iterations
        self.element_size_mm = float(element_size_mm)
        self.iterations = iterations
        self.soft_contact_margin_m = float(soft_contact_margin_m)
        self.carrier_contact_stiffness_n_m = float(
            carrier_contact_stiffness_n_m
        )
        self._has_run = False

    def run(
        self,
        *,
        inspect_trial: Callable[
            [DesignTrial, LumoSimulation, Indenter],
            None,
        ]
        | None = None,
    ) -> None:
        """Run and release each trial, optionally inspecting its live result."""
        if self._has_run:
            raise RuntimeError("design study has already been run")
        if inspect_trial is not None and not callable(inspect_trial):
            raise TypeError("inspect_trial must be callable")
        self._has_run = True

        for trial in self.trials:
            initial_tf = np.asarray(trial.initial_tf, dtype=np.float64)
            initial_translation_W_m = initial_tf[:3]
            initial_rotation = wp.quat(*initial_tf[3:])
            translation_step_W_m = np.asarray(
                trial.translation_step_W_m,
                dtype=np.float64,
            )
            position_step_m = float(np.linalg.norm(translation_step_W_m))
            motion_direction_W = translation_step_W_m / position_step_m
            max_step_count = int(
                trial.max_sim_time_s * self.sim_frequency
            )
            if max_step_count < 1:
                raise ValueError(
                    "max_sim_time_s must include at least one simulation tick"
                )
            required_force_ticks = max(
                1,
                ceil(self.force_duration_s * self.sim_frequency),
            )

            builder = newton.ModelBuilder(gravity=0.0)
            indenter = Indenter.add_urdf(
                builder,
                trial.urdf_path,
                tf=trial.initial_tf,
            )
            simulation = LumoSimulation(
                self.fingertip,
                builder=builder,
                sim_frequency=self.sim_frequency,
                iterations=self.iterations,
                soft_contact_margin_m=self.soft_contact_margin_m,
                element_size_mm=self.element_size_mm,
                carrier_contact_stiffness_n_m=(
                    self.carrier_contact_stiffness_n_m
                ),
            )

            if simulation.soft_contact_count(indenter.body_index):
                raise RuntimeError(
                    f"{trial.name} has soft contacts before prescribed motion"
                )

            travel_m = 0.0
            previous_search_error_n = -trial.target_force_n
            reaction_force_n = 0.0
            force_change_n = float("inf")
            previous_tick_force_n: float | None = None
            target_triggered = False
            force_ticks = 0
            search_iteration = 0
            pose = trial.initial_tf

            while simulation.step_count < max_step_count:
                if not target_triggered:
                    correction_sign = 1.0
                elif reaction_force_n < (
                    trial.target_force_n - self.force_tolerance_n
                ):
                    correction_sign = 1.0
                elif reaction_force_n > (
                    trial.target_force_n + self.force_tolerance_n
                ):
                    correction_sign = -1.0
                else:
                    correction_sign = 0.0

                if correction_sign != 0.0:
                    if target_triggered:
                        if search_iteration >= self.max_search_iterations:
                            raise RuntimeError(
                                f"{trial.name} did not hold "
                                f"{trial.target_force_n:g} N within tolerance "
                                f"after {self.max_search_iterations} pose "
                                "corrections"
                            )
                        search_iteration += 1
                    travel_m += correction_sign * position_step_m
                    translation_W_m = (
                        initial_translation_W_m
                        + travel_m * motion_direction_W
                    )
                    pose = wp.transform(
                        wp.vec3(*translation_W_m),
                        initial_rotation,
                    )
                    simulation.apply_indenter_pose(indenter, pose)

                simulation.step()
                reaction_force_n = simulation.indenter_reaction_force(
                    indenter,
                    motion_direction_W=trial.translation_step_W_m,
                )
                force_change_n = (
                    float("inf")
                    if previous_tick_force_n is None
                    else abs(reaction_force_n - previous_tick_force_n)
                )
                previous_tick_force_n = reaction_force_n
                search_error_n = reaction_force_n - trial.target_force_n

                if not target_triggered and reaction_force_n >= (
                    trial.target_force_n
                ):
                    target_triggered = True

                if (
                    target_triggered
                    and abs(search_error_n) <= self.force_tolerance_n
                ):
                    # The trigger or correction tick intentionally counts as
                    # the first consecutive in-band force sample.
                    force_ticks += 1
                else:
                    force_ticks = 0
                    if (
                        target_triggered
                        and search_error_n * previous_search_error_n < 0.0
                    ):
                        position_step_m *= 0.5
                    if target_triggered:
                        previous_search_error_n = search_error_n

                if force_ticks >= required_force_ticks:
                    trial.final_tf = pose
                    trial.travel_m = travel_m
                    trial.step_count = simulation.step_count
                    trial.simulation_time_s = simulation.time_s
                    trial.search_iteration_count = search_iteration
                    trial.reaction_force_n = reaction_force_n
                    trial.maximum_particle_speed_m_s = (
                        simulation.maximum_active_particle_speed_m_s()
                    )
                    trial.force_change_n = force_change_n
                    if inspect_trial is not None:
                        inspect_trial(trial, simulation, indenter)
                    break
            else:
                raise RuntimeError(
                    f"{trial.name} did not hold {trial.target_force_n:g} N "
                    f"within tolerance for {self.force_duration_s:g} s "
                    f"during {trial.max_sim_time_s:g} s of simulation; last "
                    f"force was {reaction_force_n:.9e} N"
                )

            del simulation, indenter, builder


__all__ = ["DesignStudy", "DesignTrial"]
