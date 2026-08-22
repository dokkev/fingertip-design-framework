"""Independent indentation cases sharing one analytic fingertip."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.newton import Indenter
from lumo.simulation.runtime import LumoSimulation
from lumo.util.scalar_validation import require_positive


class IndentationCase:
    """Definition and lightweight result of one URDF indentation."""

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


class IndentationStudy:
    """Run ordered independent indentation cases on one analytic fingertip."""

    def __init__(
        self,
        fingertip: Fingertip,
        cases: Iterable[IndentationCase],
        *,
        sim_frequency: float,
        force_tolerance_n: float,
        velocity_tolerance_m_s: float,
        force_change_tolerance_n: float,
        settled_tick_count: int,
        max_search_iterations: int,
    ) -> None:
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be a Fingertip")
        require_positive("sim_frequency", sim_frequency)
        require_positive("force_tolerance_n", force_tolerance_n)
        require_positive("velocity_tolerance_m_s", velocity_tolerance_m_s)
        require_positive(
            "force_change_tolerance_n",
            force_change_tolerance_n,
        )
        for name, value in (
            ("settled_tick_count", settled_tick_count),
            ("max_search_iterations", max_search_iterations),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        case_tuple = tuple(cases)
        if not case_tuple:
            raise ValueError("cases must contain at least one indentation")
        if any(
            not isinstance(case, IndentationCase)
            for case in case_tuple
        ):
            raise TypeError("cases must contain only IndentationCase objects")
        if len({id(case) for case in case_tuple}) != len(case_tuple):
            raise ValueError("cases must not contain the same object twice")
        if any(case.final_tf is not None for case in case_tuple):
            raise ValueError("cases must not have been run already")

        self.fingertip = fingertip
        self.cases = case_tuple
        self.sim_frequency = float(sim_frequency)
        self.force_tolerance_n = float(force_tolerance_n)
        self.velocity_tolerance_m_s = float(velocity_tolerance_m_s)
        self.force_change_tolerance_n = float(force_change_tolerance_n)
        self.settled_tick_count = settled_tick_count
        self.max_search_iterations = max_search_iterations
        self._has_run = False

    def run(
        self,
        *,
        inspect_case: Callable[
            [IndentationCase, LumoSimulation, Indenter],
            None,
        ]
        | None = None,
    ) -> None:
        """Run and release each case, optionally inspecting its live result."""
        if self._has_run:
            raise RuntimeError("indentation study has already been run")
        if inspect_case is not None and not callable(inspect_case):
            raise TypeError("inspect_case must be callable")
        self._has_run = True

        for case in self.cases:
            initial_tf = np.asarray(case.initial_tf, dtype=np.float64)
            initial_translation_W_m = initial_tf[:3]
            initial_rotation = wp.quat(*initial_tf[3:])
            translation_step_W_m = np.asarray(
                case.translation_step_W_m,
                dtype=np.float64,
            )
            position_step_m = float(np.linalg.norm(translation_step_W_m))
            motion_direction_W = translation_step_W_m / position_step_m
            max_step_count = int(
                case.max_sim_time_s * self.sim_frequency
            )
            if max_step_count < 1:
                raise ValueError(
                    "max_sim_time_s must include at least one simulation tick"
                )

            builder = newton.ModelBuilder(gravity=0.0)
            indenter = Indenter.add_urdf(
                builder,
                case.urdf_path,
                tf=case.initial_tf,
            )
            simulation = LumoSimulation(
                self.fingertip,
                builder=builder,
                sim_frequency=self.sim_frequency,
            )

            simulation.collision_pipeline.collide(
                simulation.state,
                simulation.contacts,
            )
            if _soft_contact_count_for_body(
                simulation,
                indenter.body_index,
            ):
                raise RuntimeError(
                    f"{case.name} has soft contacts before prescribed motion"
                )

            travel_m = 0.0
            previous_search_error_n = -case.target_force_n
            reaction_force_n = 0.0
            maximum_particle_speed_m_s = float("inf")
            force_change_n = float("inf")

            for search_iteration in range(
                1,
                self.max_search_iterations + 1,
            ):
                if previous_search_error_n < 0.0:
                    travel_m += position_step_m
                else:
                    travel_m -= position_step_m
                translation_W_m = (
                    initial_translation_W_m
                    + travel_m * motion_direction_W
                )
                pose = wp.transform(
                    wp.vec3(*translation_W_m),
                    initial_rotation,
                )
                simulation.apply_indenter_pose(
                    indenter,
                    pose,
                )

                settled_ticks = 0
                previous_tick_force_n: float | None = None
                while settled_ticks < self.settled_tick_count:
                    if simulation.step_count >= max_step_count:
                        raise RuntimeError(
                            f"{case.name} did not reach a settled "
                            f"{case.target_force_n:g} N force within "
                            f"{case.max_sim_time_s:g} s; last observed force "
                            f"was {reaction_force_n:.9e} N"
                        )

                    simulation.step()
                    reaction_force_n = simulation.indenter_reaction_force(
                        indenter,
                        motion_direction_W=case.translation_step_W_m,
                    )
                    maximum_particle_speed_m_s = (
                        simulation.maximum_active_particle_speed_m_s()
                    )
                    force_change_n = (
                        float("inf")
                        if previous_tick_force_n is None
                        else abs(reaction_force_n - previous_tick_force_n)
                    )
                    previous_tick_force_n = reaction_force_n

                    if (
                        maximum_particle_speed_m_s
                        <= self.velocity_tolerance_m_s
                        and force_change_n
                        <= self.force_change_tolerance_n
                    ):
                        settled_ticks += 1
                    else:
                        settled_ticks = 0

                search_error_n = reaction_force_n - case.target_force_n
                if abs(search_error_n) <= self.force_tolerance_n:
                    case.final_tf = pose
                    case.travel_m = travel_m
                    case.step_count = simulation.step_count
                    case.simulation_time_s = simulation.time_s
                    case.search_iteration_count = search_iteration
                    case.reaction_force_n = reaction_force_n
                    case.maximum_particle_speed_m_s = (
                        maximum_particle_speed_m_s
                    )
                    case.force_change_n = force_change_n
                    if inspect_case is not None:
                        inspect_case(case, simulation, indenter)
                    break

                if search_error_n * previous_search_error_n < 0.0:
                    position_step_m *= 0.5
                previous_search_error_n = search_error_n
            else:
                raise RuntimeError(
                    f"{case.name} did not reach the settled "
                    f"{case.target_force_n:g} N target within "
                    f"{self.max_search_iterations} search iterations; last "
                    f"settled force was {reaction_force_n:.9e} N"
                )

            del simulation, indenter, builder


def _soft_contact_count_for_body(
    simulation: LumoSimulation,
    body_index: int,
) -> int:
    contact_count = int(
        simulation.contacts.soft_contact_count.numpy()[0]
    )
    shape_indices = simulation.contacts.soft_contact_shape.numpy()[
        :contact_count
    ]
    valid = shape_indices >= 0
    shape_bodies = simulation.fingertip_model.model.shape_body.numpy()
    return int(
        np.count_nonzero(
            shape_bodies[shape_indices[valid]] == body_index
        )
    )


__all__ = ["IndentationCase", "IndentationStudy"]
