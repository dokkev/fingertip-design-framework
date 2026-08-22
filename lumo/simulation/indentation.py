"""Independent indentation cases sharing one analytic fingertip."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.newton import Indenter
from lumo.simulation.runtime import LumoSimulation
from lumo.util.scalar_validation import require_positive


class IndentationCase:
    """Definition and final state of one prescribed URDF indentation."""

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

        self.simulation: LumoSimulation | None = None
        self.indenter: Indenter | None = None
        self.step_count = 0
        self.reaction_force_n: float | None = None


class IndentationStudy:
    """Run ordered independent indentation cases on one analytic fingertip."""

    def __init__(
        self,
        fingertip: Fingertip,
        cases: Iterable[IndentationCase],
        *,
        sim_frequency: float,
    ) -> None:
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be a Fingertip")
        require_positive("sim_frequency", sim_frequency)

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
        if any(case.simulation is not None for case in case_tuple):
            raise ValueError("cases must not have been run already")

        self.fingertip = fingertip
        self.cases = case_tuple
        self.sim_frequency = float(sim_frequency)
        self._has_run = False

    def run(self) -> None:
        """Run every case from a fresh reference state, in input order."""
        if self._has_run:
            raise RuntimeError("indentation study has already been run")
        self._has_run = True

        for case in self.cases:
            initial_tf = np.asarray(case.initial_tf, dtype=np.float64)
            initial_translation_W_m = initial_tf[:3]
            initial_rotation = wp.quat(*initial_tf[3:])
            translation_step_W_m = np.asarray(
                case.translation_step_W_m,
                dtype=np.float64,
            )
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
            case.simulation = simulation
            case.indenter = indenter

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

            for step_count in range(1, max_step_count + 1):
                translation_W_m = (
                    initial_translation_W_m
                    + step_count * translation_step_W_m
                )
                simulation.apply_indenter_pose(
                    indenter,
                    wp.transform(
                        wp.vec3(*translation_W_m),
                        initial_rotation,
                    ),
                )
                simulation.step()
                case.step_count = step_count
                case.reaction_force_n = simulation.indenter_reaction_force(
                    indenter,
                    motion_direction_W=case.translation_step_W_m,
                )
                if case.reaction_force_n >= case.target_force_n:
                    break
            else:
                raise RuntimeError(
                    f"{case.name} did not reach {case.target_force_n:g} N "
                    f"within {case.max_sim_time_s:g} s; last force was "
                    f"{case.reaction_force_n:.9e} N"
                )


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
