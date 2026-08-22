"""Prescribed indentation cases for LUMO simulations."""

from __future__ import annotations

import numpy as np
import warp as wp

from lumo.newton import Indenter
from lumo.simulation import LumoSimulation
from lumo.util.scalar_validation import require_positive


class IndentationCase:
    """One translated indenter stopped by a transient reaction-force target."""

    def __init__(
        self,
        simulation: LumoSimulation,
        indenter: Indenter,
        *,
        name: str,
        initial_tf: wp.transform,
        translation_step_W_m: wp.vec3,
        target_force_n: float,
        max_sim_time_s: float,
    ) -> None:
        if not isinstance(simulation, LumoSimulation):
            raise TypeError("simulation must be a LumoSimulation")
        if not isinstance(indenter, Indenter):
            raise TypeError("indenter must be an Indenter")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty string")
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

        max_step_count = int(
            float(max_sim_time_s) * simulation.sim_frequency
        )
        if max_step_count < 1:
            raise ValueError(
                "max_sim_time_s must include at least one simulation tick"
            )

        self.simulation = simulation
        self.indenter = indenter
        self.name = name.strip()
        self.target_force_n = float(target_force_n)
        self.max_sim_time_s = float(max_sim_time_s)
        self._initial_translation_W_m = initial_tf_values[:3].copy()
        self._initial_rotation = wp.quat(*initial_tf_values[3:])
        self._translation_step_W_m = translation_step
        self._translation_step_norm_m = translation_step_norm_m
        self._motion_direction_W = wp.vec3(*translation_step)
        self._max_step_count = max_step_count
        self._step_count = 0
        self._reaction_force_n = 0.0
        self._prepared_simulation_step: int | None = None

    @property
    def target_reached(self) -> bool:
        """Whether the most recent transient force reached the target."""
        return self._reaction_force_n >= self.target_force_n

    @property
    def step_count(self) -> int:
        """Number of completed simulation ticks in this case."""
        return self._step_count

    @property
    def elapsed_time_s(self) -> float:
        """Completed case time derived from the simulation frequency."""
        return self._step_count / self.simulation.sim_frequency

    @property
    def travel_m(self) -> float:
        """Prescribed translation distance completed by this case."""
        return self._step_count * self._translation_step_norm_m

    @property
    def reaction_force_n(self) -> float:
        """Most recently observed transient reaction force."""
        return self._reaction_force_n

    def apply_next_pose(self) -> None:
        """Apply the next prescribed pose without advancing simulation time."""
        if self._prepared_simulation_step is not None:
            raise RuntimeError(
                "observe_step() must follow apply_next_pose()"
            )
        if self.target_reached:
            raise RuntimeError(f"{self.name} already reached its force target")
        if self._step_count >= self._max_step_count:
            raise RuntimeError(
                f"{self.name} did not reach {self.target_force_n:g} N "
                f"within {self.max_sim_time_s:g} s; last force was "
                f"{self._reaction_force_n:.9e} N"
            )

        next_translation_W_m = (
            self._initial_translation_W_m
            + (self._step_count + 1) * self._translation_step_W_m
        )
        self.simulation.apply_indenter_pose(
            self.indenter,
            wp.transform(
                wp.vec3(*next_translation_W_m),
                self._initial_rotation,
            ),
        )
        self._prepared_simulation_step = self.simulation.step_count

    def observe_step(self) -> float:
        """Observe exactly one simulation tick after the prepared pose."""
        if self._prepared_simulation_step is None:
            raise RuntimeError(
                "apply_next_pose() must precede observe_step()"
            )
        if self.simulation.step_count != self._prepared_simulation_step + 1:
            raise RuntimeError(
                "exactly one LumoSimulation.step() must occur between "
                "apply_next_pose() and observe_step()"
            )

        self._reaction_force_n = (
            self.simulation.indenter_reaction_force(
                self.indenter,
                motion_direction_W=self._motion_direction_W,
            )
        )
        self._step_count += 1
        self._prepared_simulation_step = None
        return self._reaction_force_n


__all__ = ["IndentationCase"]
