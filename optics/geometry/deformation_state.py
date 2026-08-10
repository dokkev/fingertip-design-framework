"""Replaceable displacement state for one fixed optical mesh topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from optics.geometry.pad_mesh_template import PadMeshTemplate2D


class InvalidPadDeformationState(ValueError):
    """Raised when a pad displacement field is malformed."""


@dataclass(frozen=True)
class PadDeformationState2D:
    """Nodal x-y displacement in millimeters for one load condition."""

    displacement_mm: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Own immutable copies of displacement and metadata."""
        displacement = np.array(self.displacement_mm, dtype=float, copy=True)
        if displacement.ndim != 2 or displacement.shape[1:] != (2,):
            raise InvalidPadDeformationState(
                "displacement_mm must have shape (N, 2)"
            )
        if not np.all(np.isfinite(displacement)):
            raise InvalidPadDeformationState(
                "displacement_mm must contain only finite values"
            )
        displacement.setflags(write=False)
        object.__setattr__(self, "displacement_mm", displacement)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @classmethod
    def zero(
        cls,
        template: PadMeshTemplate2D,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> PadDeformationState2D:
        """Return the no-load state for ``template``."""
        state_metadata = {"condition": "no_load"}
        if metadata is not None:
            state_metadata.update(metadata)
        return cls(
            displacement_mm=np.zeros_like(template.reference_coordinates_mm),
            metadata=state_metadata,
        )


@dataclass(frozen=True)
class PadField2D:
    """One fixed pad topology paired with one replaceable load state."""

    template: PadMeshTemplate2D
    state: PadDeformationState2D

    def __post_init__(self) -> None:
        self.template.validate_state(self.state)

    @property
    def deformed_coordinates_mm(self) -> np.ndarray:
        """Return reference coordinates plus the current displacement."""
        return self.template.coordinates_for(self.state)

    @property
    def zero_state(self) -> PadDeformationState2D:
        """Return a no-load state on the same topology."""
        return PadDeformationState2D.zero(self.template)
