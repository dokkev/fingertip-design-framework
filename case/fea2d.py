"""Configuration and result ownership for one 2D explicit-contact solve."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fem import FEAResult, solve as solve_explicit_contact
from mesh import MeshSettings, mesh_settings_for_level
from mesh.indenter import IndenterSettings
from model import Fingertip

from case.state import ContactState


@dataclass
class FEA2D:
    """One 2D mechanics experiment and its optional result."""

    contact: ContactState
    indenter: IndenterSettings = field(default_factory=IndenterSettings)
    mesh_settings: MeshSettings = field(
        default_factory=lambda: mesh_settings_for_level("medium")
    )
    steps: int = 48
    internal_contact: str = "three_pairs"
    result: FEAResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.indenter, IndenterSettings):
            raise TypeError("indenter must be IndenterSettings")
        if not isinstance(self.contact, ContactState):
            raise TypeError("contact must be ContactState")
        if not isinstance(self.mesh_settings, MeshSettings):
            raise TypeError("mesh_settings must be MeshSettings")
        if self.contact.indenter_radius_mm != self.indenter.radius_mm:
            raise ValueError("contact indenter radius must match indenter settings")
        if (
            not isinstance(self.steps, int)
            or isinstance(self.steps, bool)
            or self.steps <= 0
        ):
            raise ValueError("steps must be a positive integer")
        if not isinstance(self.internal_contact, str) or not self.internal_contact:
            raise ValueError("internal_contact must be a nonempty string")

    def solve(self, fingertip: Fingertip) -> FEAResult:
        """Solve this configuration against the supplied physical fingertip."""
        if not isinstance(fingertip, Fingertip):
            raise TypeError("fingertip must be Fingertip")
        mesh = fingertip.mesh(self.mesh_settings)
        self.result = solve_explicit_contact(
            fingertip,
            mesh,
            indentation=self.contact.indentation_mm,
            surface_x_mm=self.contact.location_x_mm,
            steps=self.steps,
            indenter=self.indenter,
            internal_contact=self.internal_contact,
        )
        return self.result


__all__ = ["FEA2D"]
