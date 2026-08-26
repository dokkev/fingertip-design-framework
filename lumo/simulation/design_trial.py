"""Design trials that evaluate one fingertip design."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import ceil
from pathlib import Path

import newton
import numpy as np
import warp as wp

from lumo.fingertip import Fingertip
from lumo.mesh import FingertipMesh
from lumo.newton import Indenter
from lumo.simulation.runtime import LumoSimulation
from lumo.util.scalar_validation import require_nonnegative, require_positive


_DEFAULT_FORCE_GAIN_M_S_N = 1.25e-3
REFERENCE_DWELL_LOADING = "reference_dwell"
QUASISTATIC_RAMP_LOADING = "quasistatic_ramp"


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
        self._has_run = False

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
        if loading_mode not in {REFERENCE_DWELL_LOADING, QUASISTATIC_RAMP_LOADING}:
            raise ValueError(
                "loading_mode must be 'reference_dwell' or 'quasistatic_ramp'"
            )
        if loading_mode == REFERENCE_DWELL_LOADING:
            require_positive("settle_duration_s", self.settle_duration_s)
            if force_ramp_rate_n_s is not None:
                raise ValueError(
                    "force_ramp_rate_n_s is only valid for quasistatic_ramp"
                )
        else:
            if force_ramp_rate_n_s is None:
                raise ValueError(
                    "quasistatic_ramp requires force_ramp_rate_n_s"
                )
            require_positive("force_ramp_rate_n_s", force_ramp_rate_n_s)
        self._has_run = True

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
            settle_ticks = max(
                1,
                ceil(self.settle_duration_s * self.sim_frequency),
            )

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
                use_cuda_graph=self.use_cuda_graph,
            )

            if simulation.soft_contact_count(indenter.body_index):
                raise RuntimeError(
                    f"{trial.name} has soft contacts before prescribed motion"
                )

            travel_m = 0.0
            reaction_force_n = 0.0
            force_change_n = float("inf")
            previous_force_n = 0.0
            in_tolerance_ticks = 0
            pose = trial.initial_tf
            force_targets_n = self.force_targets_n or (trial.target_force_n,)
            target_index = 0
            target_force_n = force_targets_n[target_index]
            force_tolerance_n = (
                self.force_tolerance_n
                if self.force_tolerance_n is not None
                else target_force_n * self.force_tolerance_fraction
            )

            while simulation.step_count < max_step_count:
                force_reference_n = target_force_n
                if loading_mode == QUASISTATIC_RAMP_LOADING:
                    force_reference_n = min(
                        force_targets_n[-1],
                        force_ramp_rate_n_s * simulation.time_s,
                    )
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

                if loading_mode == QUASISTATIC_RAMP_LOADING:
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
                force_tolerance_n = (
                    self.force_tolerance_n
                    if self.force_tolerance_n is not None
                    else target_force_n * self.force_tolerance_fraction
                )
                in_tolerance_ticks = 0
            else:
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

            del simulation, indenter, builder


__all__ = [
    "DesignStudy",
    "DesignTrial",
    "QUASISTATIC_RAMP_LOADING",
    "REFERENCE_DWELL_LOADING",
]
