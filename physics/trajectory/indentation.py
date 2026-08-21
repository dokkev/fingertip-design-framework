"""Solver-neutral rigid-object indentation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from mesh.rigid.carrier import RigidCarrierMesh
from mesh.rigid.object import RigidObjectMesh, RigidPose3D

from .fingertip import PreparedFingertipMesh
from ..contracts.types import NewtonResult
from ..newton.solve import NewtonSettings, _load_newton_backend

if TYPE_CHECKING:
    from contact.first_contact import FirstContactResult


DIRECTION_NORM_TOLERANCE = 1.0e-12


class CandidateMechanicsError(RuntimeError):
    """Raised when one candidate produces an unacceptable mechanics state."""


@dataclass(frozen=True)
class CheckpointStep:
    """One named step in an exact prescribed-travel schedule."""

    travel_mm: float
    interval_step: int
    cumulative_step: int


def _finite_tuple(
    value: tuple[float, ...] | list[float],
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(component) for component in array)


def _normalize(
    value: tuple[float, ...],
    *,
    tolerance: float,
    name: str,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= tolerance:
        raise ValueError(f"{name} must have a finite nonzero norm")
    return tuple(float(component) for component in array / norm)


@dataclass(frozen=True)
class RigidIndenter3D:
    """A neutral rigid mesh, initial pose, and prescribed approach direction."""

    mesh: RigidObjectMesh
    initial_pose: RigidPose3D
    approach_direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, RigidObjectMesh):
            raise TypeError("mesh must be a RigidObjectMesh")
        if not isinstance(self.initial_pose, RigidPose3D):
            raise TypeError("initial_pose must be a RigidPose3D")
        direction = _finite_tuple(
            self.approach_direction,
            length=3,
            name="approach_direction",
        )
        object.__setattr__(
            self,
            "approach_direction",
            _normalize(
                direction,
                tolerance=DIRECTION_NORM_TOLERANCE,
                name="approach_direction",
            ),
        )

    def pose_at_travel(self, travel_mm: float) -> RigidPose3D:
        """Return the translated pose at nonnegative prescribed travel."""

        travel = float(travel_mm)
        if not np.isfinite(travel) or travel < 0.0:
            raise ValueError("travel_mm must be finite and non-negative")
        translation = np.asarray(self.initial_pose.translation_mm, dtype=float)
        translation += travel * np.asarray(self.approach_direction, dtype=float)
        return RigidPose3D(
            translation_mm=tuple(float(component) for component in translation),
            quaternion_xyzw=self.initial_pose.quaternion_xyzw,
        )


@dataclass(frozen=True)
class IndentationSettings:
    """Minimal contact execution settings for one translation-only protocol."""

    travel_mm: float
    load_steps: int = 8
    soft_contact_margin_mm: float = 0.05
    """Soft contact detection margin in repository millimetres."""
    rigid_sdf_target_voxel_mm: float = 0.125
    """Explicit contact-scale voxel size for the rigid mesh SDF in mm."""
    # Newton VBD wiring defaults, not calibrated silicone properties.
    soft_contact_ke: float = 1.0e3
    soft_contact_kd: float = 10.0
    soft_contact_mu: float = 0.0

    def __post_init__(self) -> None:
        travel = float(self.travel_mm)
        if not np.isfinite(travel) or travel < 0.0:
            raise ValueError("travel_mm must be finite and non-negative")
        if int(self.load_steps) != self.load_steps or int(self.load_steps) < 1:
            raise ValueError("load_steps must be a positive integer")
        for name in (
            "soft_contact_margin_mm",
            "rigid_sdf_target_voxel_mm",
            "soft_contact_ke",
            "soft_contact_kd",
            "soft_contact_mu",
        ):
            value = float(getattr(self, name))
            lower_bound = 0.0 if name != "rigid_sdf_target_voxel_mm" else 1.0e-12
            if not np.isfinite(value) or value < lower_bound:
                qualifier = "positive" if name == "rigid_sdf_target_voxel_mm" else "finite and non-negative"
                raise ValueError(f"{name} must be {qualifier}")
        object.__setattr__(self, "travel_mm", travel)
        object.__setattr__(self, "load_steps", int(self.load_steps))
        object.__setattr__(self, "soft_contact_margin_mm", float(self.soft_contact_margin_mm))
        object.__setattr__(self, "rigid_sdf_target_voxel_mm", float(self.rigid_sdf_target_voxel_mm))
        object.__setattr__(self, "soft_contact_ke", float(self.soft_contact_ke))
        object.__setattr__(self, "soft_contact_kd", float(self.soft_contact_kd))
        object.__setattr__(self, "soft_contact_mu", float(self.soft_contact_mu))


@dataclass(frozen=True)
class IndentationResult:
    """Neutral fingertip result plus the final prescribed rigid-object pose."""

    mechanics_result: NewtonResult
    final_indenter_pose: RigidPose3D
    checkpoint_state: "MechanicsCheckpointState | None" = None
    diagnostics: Mapping[str, float | int | str | bool | tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mechanics_result, NewtonResult):
            raise TypeError("mechanics_result must be a NewtonResult")
        if not isinstance(self.final_indenter_pose, RigidPose3D):
            raise TypeError("final_indenter_pose must be a RigidPose3D")
        if self.checkpoint_state is not None and not isinstance(
            self.checkpoint_state, MechanicsCheckpointState
        ):
            raise TypeError("checkpoint_state must be MechanicsCheckpointState or None")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class MechanicsCheckpointState:
    """Validated mechanics state required by downstream simulation stages."""

    active_carrier_contact_vertex_indices: tuple[int, ...]
    rigid_sdf_target_voxel_mm: float
    final_pose_error_mm: float
    carrier_contact_active: bool
    carrier_contact_occurred: bool
    first_carrier_contact_step: int | None
    carrier_collision_enabled: bool
    inverted_tetrahedra: int
    max_soft_contact_overflow: int
    max_rigid_contact_overflow: int
    max_support_displacement_mm: float
    max_carrier_penetration_mm: float
    carrier_interface_contact_count: int
    first_contact_travel_mm: float | None = None
    spawn_clearance_mm: float | None = None

    @classmethod
    def from_diagnostics(
        cls,
        diagnostics: Mapping[str, object],
    ) -> "MechanicsCheckpointState":
        """Extract and validate the fixed downstream contract inside physics."""

        required = (
            "active_carrier_contact_vertex_indices",
            "rigid_sdf_target_voxel_mm",
            "final_pose_error_mm",
            "carrier_contact_active",
            "carrier_contact_occurred",
            "first_carrier_contact_step",
            "carrier_collision_enabled",
            "inverted_tetrahedra",
            "max_soft_contact_overflow",
            "max_rigid_contact_overflow",
            "max_support_displacement_mm",
            "max_carrier_penetration_mm",
            "carrier_interface_contact_count",
        )
        missing = tuple(name for name in required if name not in diagnostics)
        if missing:
            raise ValueError(
                "mechanics checkpoint is missing required state: " f"{missing!r}"
            )
        raw_indices = np.asarray(diagnostics["active_carrier_contact_vertex_indices"])
        if raw_indices.ndim != 1 or (
            raw_indices.size and not np.issubdtype(raw_indices.dtype, np.integer)
        ):
            raise ValueError(
                "active_carrier_contact_vertex_indices must be a 1D integer sequence"
            )
        indices = tuple(int(index) for index in raw_indices)
        if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError(
                "active_carrier_contact_vertex_indices must be unique and nonnegative"
            )

        def finite_nonnegative(name: str) -> float:
            value = float(diagnostics[name])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            return value

        def nonnegative_int(name: str) -> int:
            value = diagnostics[name]
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            return int(value)

        def strict_bool(name: str) -> bool:
            value = diagnostics[name]
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
            return value

        first_step = diagnostics["first_carrier_contact_step"]
        if first_step is not None:
            if (
                isinstance(first_step, bool)
                or int(first_step) != first_step
                or int(first_step) < 1
            ):
                raise ValueError("first_carrier_contact_step must be None or positive")
            first_step = int(first_step)
        return cls(
            active_carrier_contact_vertex_indices=indices,
            rigid_sdf_target_voxel_mm=finite_nonnegative("rigid_sdf_target_voxel_mm"),
            final_pose_error_mm=finite_nonnegative("final_pose_error_mm"),
            carrier_contact_active=strict_bool("carrier_contact_active"),
            carrier_contact_occurred=strict_bool("carrier_contact_occurred"),
            first_carrier_contact_step=first_step,
            carrier_collision_enabled=strict_bool("carrier_collision_enabled"),
            inverted_tetrahedra=nonnegative_int("inverted_tetrahedra"),
            max_soft_contact_overflow=nonnegative_int("max_soft_contact_overflow"),
            max_rigid_contact_overflow=nonnegative_int("max_rigid_contact_overflow"),
            max_support_displacement_mm=finite_nonnegative("max_support_displacement_mm"),
            max_carrier_penetration_mm=finite_nonnegative("max_carrier_penetration_mm"),
            carrier_interface_contact_count=nonnegative_int(
                "carrier_interface_contact_count"
            ),
            first_contact_travel_mm=(
                None
                if diagnostics.get("first_contact_travel_mm") is None
                else finite_nonnegative("first_contact_travel_mm")
            ),
            spawn_clearance_mm=(
                None
                if diagnostics.get("spawn_clearance_mm") is None
                else finite_nonnegative("spawn_clearance_mm")
            ),
        )


@dataclass(frozen=True)
class IndentationCheckpoint:
    """Immutable snapshot of one accepted point on an indentation path."""

    checkpoint_index: int
    checkpoint_fraction: float
    normalized_indentation_ratio: float
    post_contact_travel_mm: float
    cumulative_step_index: int
    indenter_pose: RigidPose3D
    mechanics_result: NewtonResult
    state: MechanicsCheckpointState
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.indenter_pose, RigidPose3D):
            raise TypeError("indenter_pose must be RigidPose3D")
        if not isinstance(self.mechanics_result, NewtonResult):
            raise TypeError("mechanics_result must be NewtonResult")
        if not isinstance(self.state, MechanicsCheckpointState):
            raise TypeError("state must be MechanicsCheckpointState")
        for name in (
            "checkpoint_fraction",
            "normalized_indentation_ratio",
            "post_contact_travel_mm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if int(self.checkpoint_index) < 0 or int(self.cumulative_step_index) < 1:
            raise ValueError("checkpoint and cumulative step indices are invalid")
        object.__setattr__(self, "checkpoint_index", int(self.checkpoint_index))
        object.__setattr__(self, "cumulative_step_index", int(self.cumulative_step_index))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class IndentationTrajectoryResult:
    """All copied checkpoints from one continuous Newton indentation path."""

    checkpoints: tuple[IndentationCheckpoint, ...]
    total_steps: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        checkpoints = tuple(self.checkpoints)
        if not checkpoints:
            raise ValueError("an indentation trajectory needs at least one checkpoint")
        if any(not isinstance(item, IndentationCheckpoint) for item in checkpoints):
            raise TypeError("checkpoints must contain IndentationCheckpoint values")
        if any(left.cumulative_step_index >= right.cumulative_step_index
               for left, right in zip(checkpoints, checkpoints[1:])):
            raise ValueError("checkpoint step indices must be strictly increasing")
        if int(self.total_steps) != checkpoints[-1].cumulative_step_index:
            raise ValueError("total_steps must equal the final checkpoint step")
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "total_steps", int(self.total_steps))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def final(self) -> IndentationCheckpoint:
        return self.checkpoints[-1]


def _validate_support_constraints(
    prepared_fingertip: PreparedFingertipMesh,
    mechanics_settings: NewtonSettings,
) -> None:
    """Require explicit mechanics supports to match prepared geometry supports."""

    configured_support = tuple(sorted(mechanics_settings.fixed_vertex_indices))
    authoritative_support = tuple(sorted(prepared_fingertip.support_vertex_indices))
    if configured_support != authoritative_support:
        raise ValueError(
            "physics indentation requires fixed_vertex_indices to exactly equal "
            "prepared_fingertip.support_vertex_indices; the authoritative "
            f"support is {authoritative_support!r}, received {configured_support!r}"
        )


def checkpoint_step_schedule(
    checkpoint_travels_mm: Sequence[float],
    *,
    max_load_increment_mm: float,
) -> tuple[CheckpointStep, ...]:
    """Return exact cumulative steps for a monotonic checkpoint path.

    The interval is split with ``ceil(distance / max_increment)`` and
    therefore always lands exactly on the requested checkpoint.
    """

    travels = tuple(float(value) for value in checkpoint_travels_mm)
    increment_limit = float(max_load_increment_mm)
    if not travels or any(not np.isfinite(value) or value <= 0.0 for value in travels):
        raise ValueError("checkpoint travels must be finite and positive")
    if any(left >= right for left, right in zip(travels, travels[1:])):
        raise ValueError("checkpoint travels must be strictly increasing")
    if not np.isfinite(increment_limit) or increment_limit <= 0.0:
        raise ValueError("max_load_increment_mm must be finite and positive")
    schedule: list[CheckpointStep] = []
    previous = 0.0
    cumulative = 0
    for target in travels:
        interval = target - previous
        interval_steps = max(1, int(np.ceil(interval / increment_limit)))
        actual_increment = interval / interval_steps
        if actual_increment > increment_limit + 1.0e-12:
            raise ValueError("checkpoint scheduler exceeded max_load_increment_mm")
        for interval_step in range(1, interval_steps + 1):
            cumulative += 1
            current = target if interval_step == interval_steps else previous + actual_increment * interval_step
            schedule.append(
                CheckpointStep(float(current), interval_step, cumulative)
            )
        previous = target
    return tuple(schedule)


def solve_fingertip_indentation(
    prepared_fingertip: PreparedFingertipMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: NewtonSettings | None = None,
    indentation_settings: IndentationSettings | None = None,
    *,
    first_contact: FirstContactResult | None = None,
    visual_carrier_mesh: RigidObjectMesh | None = None,
    rigid_carrier_mesh: RigidCarrierMesh | None = None,
) -> IndentationResult:
    """Run one Newton VBD rigid-soft indentation through the neutral boundary.

    When ``first_contact`` is supplied, the backend starts from its verified
    collision-free ``spawn_pose`` and schedules loading relative to
    ``contact_pose``.  The first-contact object is solver-neutral and is only
    used to define the mechanical initialization convention.  A
    ``visual_carrier_mesh`` is render-only; ``rigid_carrier_mesh`` is static
    collision-enabled geometry and is deliberately a separate argument.
    """

    if not isinstance(prepared_fingertip, PreparedFingertipMesh):
        raise TypeError("prepared_fingertip must be a PreparedFingertipMesh")
    if not isinstance(indenter, RigidIndenter3D):
        raise TypeError("indenter must be a RigidIndenter3D")
    if mechanics_settings is None:
        mechanics_settings = NewtonSettings()
    if not isinstance(mechanics_settings, NewtonSettings):
        raise TypeError("mechanics_settings must be NewtonSettings")
    if indentation_settings is None:
        raise TypeError("indentation_settings must be provided")
    if not isinstance(indentation_settings, IndentationSettings):
        raise TypeError("indentation_settings must be IndentationSettings")
    if first_contact is not None:
        from contact.first_contact import FirstContactResult

        if not isinstance(first_contact, FirstContactResult):
            raise TypeError("first_contact must be FirstContactResult or None")
        if not np.allclose(
            np.asarray(first_contact.approach_direction, dtype=float),
            np.asarray(indenter.approach_direction, dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "first_contact.approach_direction must match the indenter "
                "approach_direction"
            )
    _validate_support_constraints(prepared_fingertip, mechanics_settings)

    backend = _load_newton_backend()
    return backend.solve_newton_vbd_indentation(
        prepared_fingertip,
        indenter,
        mechanics_settings,
        indentation_settings,
        visual_carrier_mesh=visual_carrier_mesh,
        rigid_carrier_mesh=rigid_carrier_mesh,
        first_contact=first_contact,
    )


def solve_fingertip_indentation_trajectory(
    prepared_fingertip: PreparedFingertipMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: NewtonSettings | None,
    indentation_settings: IndentationSettings,
    checkpoint_travels_mm: Sequence[float],
    *,
    checkpoint_fractions: Sequence[float] | None = None,
    normalized_indentation_ratios: Sequence[float] | None = None,
    max_load_increment_mm: float = 0.05,
    first_contact: FirstContactResult | None = None,
    visual_carrier_mesh: RigidObjectMesh | None = None,
    rigid_carrier_mesh: RigidCarrierMesh | None = None,
) -> IndentationTrajectoryResult:
    """Solve one continuous path and capture exact requested checkpoints."""

    if mechanics_settings is None:
        mechanics_settings = NewtonSettings()
    if not isinstance(mechanics_settings, NewtonSettings):
        raise TypeError("mechanics_settings must be NewtonSettings")
    if not isinstance(indentation_settings, IndentationSettings):
        raise TypeError("indentation_settings must be IndentationSettings")
    _validate_support_constraints(prepared_fingertip, mechanics_settings)
    travels = tuple(float(value) for value in checkpoint_travels_mm)
    fractions = (
        tuple(float(value) for value in checkpoint_fractions)
        if checkpoint_fractions is not None
        else tuple(value / travels[-1] for value in travels)
    )
    if len(fractions) != len(travels):
        raise ValueError("checkpoint_fractions must match checkpoint_travels_mm")
    backend = _load_newton_backend()
    return backend.solve_newton_vbd_indentation_trajectory(
        prepared_fingertip,
        indenter,
        mechanics_settings,
        indentation_settings,
        travels,
        checkpoint_fractions=fractions,
        normalized_indentation_ratios=(
            None
            if normalized_indentation_ratios is None
            else tuple(float(value) for value in normalized_indentation_ratios)
        ),
        max_load_increment_mm=max_load_increment_mm,
        visual_carrier_mesh=visual_carrier_mesh,
        rigid_carrier_mesh=rigid_carrier_mesh,
        first_contact=first_contact,
    )


__all__ = [
    "CandidateMechanicsError",
    "CheckpointStep",
    "IndentationResult",
    "MechanicsCheckpointState",
    "IndentationCheckpoint",
    "IndentationTrajectoryResult",
    "IndentationSettings",
    "RigidIndenter3D",
    "RigidPose3D",
    "checkpoint_step_schedule",
    "solve_fingertip_indentation",
    "solve_fingertip_indentation_trajectory",
]
