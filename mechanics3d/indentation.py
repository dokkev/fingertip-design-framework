"""Solver-neutral rigid-object indentation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import numpy as np

from mesh.rigid_object import RigidObjectMesh

from .fingertip import FingertipMechanicsMesh
from .solve import Mechanics3DSettings
from .types import Mechanics3DResult

if TYPE_CHECKING:
    from contact.first_contact import FirstContactResult


_QUATERNION_NORM_TOLERANCE = 1.0e-12
_DIRECTION_NORM_TOLERANCE = 1.0e-12


def _finite_tuple(value: tuple[float, ...] | list[float], *, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(component) for component in array)


def _normalize(value: tuple[float, ...], *, tolerance: float, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= tolerance:
        raise ValueError(f"{name} must have a finite nonzero norm")
    return tuple(float(component) for component in array / norm)


@dataclass(frozen=True)
class RigidPose3D:
    """Rigid-object pose in repository millimetres and quaternion ``xyzw`` order."""

    translation_mm: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        translation = _finite_tuple(self.translation_mm, length=3, name="translation_mm")
        quaternion = _finite_tuple(self.quaternion_xyzw, length=4, name="quaternion_xyzw")
        quaternion = _normalize(
            quaternion,
            tolerance=_QUATERNION_NORM_TOLERANCE,
            name="quaternion_xyzw",
        )
        object.__setattr__(self, "translation_mm", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)


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
                tolerance=_DIRECTION_NORM_TOLERANCE,
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

    mechanics_result: Mechanics3DResult
    final_indenter_pose: RigidPose3D
    diagnostics: Mapping[str, float | int | str | bool | tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mechanics_result, Mechanics3DResult):
            raise TypeError("mechanics_result must be a Mechanics3DResult")
        if not isinstance(self.final_indenter_pose, RigidPose3D):
            raise TypeError("final_indenter_pose must be a RigidPose3D")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def solve_fingertip_indentation(
    prepared_fingertip: FingertipMechanicsMesh,
    indenter: RigidIndenter3D,
    mechanics_settings: Mechanics3DSettings | None = None,
    indentation_settings: IndentationSettings | None = None,
    *,
    first_contact: FirstContactResult | None = None,
    visual_carrier_mesh: RigidObjectMesh | None = None,
    rigid_carrier_mesh: RigidObjectMesh | None = None,
) -> IndentationResult:
    """Run one Newton VBD rigid-soft indentation through the neutral boundary.

    When ``first_contact`` is supplied, the backend starts from its verified
    collision-free ``spawn_pose`` and schedules loading relative to
    ``contact_pose``.  The first-contact object is solver-neutral and is only
    used to define the mechanical initialization convention.  A
    ``visual_carrier_mesh`` is render-only; ``rigid_carrier_mesh`` is static
    collision-enabled geometry and is deliberately a separate argument.
    """

    if not isinstance(prepared_fingertip, FingertipMechanicsMesh):
        raise TypeError("prepared_fingertip must be a FingertipMechanicsMesh")
    if not isinstance(indenter, RigidIndenter3D):
        raise TypeError("indenter must be a RigidIndenter3D")
    if mechanics_settings is None:
        mechanics_settings = Mechanics3DSettings()
    if not isinstance(mechanics_settings, Mechanics3DSettings):
        raise TypeError("mechanics_settings must be Mechanics3DSettings")
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
    configured_support = tuple(sorted(mechanics_settings.fixed_vertex_indices))
    authoritative_support = tuple(sorted(prepared_fingertip.support_vertex_indices))
    if configured_support and configured_support != authoritative_support:
        raise ValueError(
            "mechanics3d indentation requires fixed_vertex_indices to be empty or "
            "equal to prepared_fingertip.support_vertex_indices; the authoritative "
            f"support is {authoritative_support!r}, received {configured_support!r}"
        )

    from .backends.newton_vbd import solve_newton_vbd_indentation

    return solve_newton_vbd_indentation(
        prepared_fingertip,
        indenter,
        mechanics_settings,
        indentation_settings,
        visual_carrier_mesh=visual_carrier_mesh,
        rigid_carrier_mesh=rigid_carrier_mesh,
        first_contact=first_contact,
    )


__all__ = [
    "IndentationResult",
    "IndentationSettings",
    "RigidIndenter3D",
    "RigidPose3D",
    "solve_fingertip_indentation",
]
