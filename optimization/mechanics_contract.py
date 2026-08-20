"""Frozen numerical mechanics settings for the production trajectory path."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MechanicsContract:
    """Numerical VBD settings, kept separate from physical protocol conditions."""

    sphere_subdivisions: int = 3
    max_load_increment_mm: float = 0.05
    vbd_iterations: int = 10
    dt_s: float = 1.0e-3
    soft_contact_margin_mm: float = 0.02
    soft_contact_ke: float = 1.0e3
    soft_contact_kd: float = 10.0

    def __post_init__(self) -> None:
        if self.sphere_subdivisions < 1 or self.vbd_iterations < 1:
            raise ValueError("mechanics iteration settings must be positive")
        if self.max_load_increment_mm <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("mechanics step settings must be positive")
        if self.soft_contact_margin_mm < 0.0 or self.soft_contact_ke < 0.0 or self.soft_contact_kd < 0.0:
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
