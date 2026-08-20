"""Frozen numerical mechanics settings for the production trajectory path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from contact import DEFAULT_FIRST_CONTACT_SETTINGS, FirstContactSettings


@dataclass(frozen=True)
class MechanicsContract:
    """Solver, contact, and checkpoint-acceptance settings.

    Fingertip material and inertial inputs belong to
    :class:`finger.fingertip_parameters.ViscoelasticParameters`; this contract
    only describes how the mechanics solver executes and how its result is
    accepted.
    """

    sphere_subdivisions: int = 3
    max_load_increment_mm: float = 0.05
    vbd_iterations: int = 10
    dt_s: float = 1.0e-3
    soft_contact_margin_mm: float = 0.02
    soft_contact_ke: float = 1.0e3
    soft_contact_kd: float = 10.0
    max_support_displacement_mm: float = 1.0e-9
    max_final_pose_error_mm: float = 1.0e-6
    max_carrier_penetration_voxel_fraction: float = 0.5
    first_contact: FirstContactSettings = field(
        default_factory=lambda: DEFAULT_FIRST_CONTACT_SETTINGS
    )

    def __post_init__(self) -> None:
        if not isinstance(self.first_contact, FirstContactSettings):
            raise TypeError("first_contact must be FirstContactSettings")
        for name, value in (
            ("sphere_subdivisions", self.sphere_subdivisions),
            ("vbd_iterations", self.vbd_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_load_increment_mm",
            "dt_s",
            "soft_contact_margin_mm",
            "soft_contact_ke",
            "soft_contact_kd",
            "max_support_displacement_mm",
            "max_final_pose_error_mm",
            "max_carrier_penetration_voxel_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.max_load_increment_mm <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("mechanics iteration settings must be positive")
        if (
            self.soft_contact_margin_mm < 0.0
            or self.soft_contact_ke < 0.0
            or self.soft_contact_kd < 0.0
            or self.max_support_displacement_mm < 0.0
            or self.max_final_pose_error_mm < 0.0
            or self.max_carrier_penetration_voxel_fraction < 0.0
        ):
            raise ValueError("contact settings must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


DEFAULT_MECHANICS_CONTRACT = MechanicsContract()


__all__ = ["DEFAULT_MECHANICS_CONTRACT", "MechanicsContract"]
