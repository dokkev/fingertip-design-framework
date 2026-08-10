"""Public neutral facade for fingertip indentation solves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from fem.indentation import (
    ConvergedIndentationStep,
    IndentationSettings,
    run_indentation_case,
)
from mesh.indenter import IndenterSettings
from mesh.pad import PadMesh
from mesh.types import FingertipMesh
from model import Fingertip


@dataclass(frozen=True)
class FEAResult:
    """Neutral final state of one displacement-controlled indentation solve."""

    mesh: PadMesh
    displacement: np.ndarray | None
    reaction_force: float | None
    contact: Mapping[str, Any]
    converged: bool
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.reaction_force is not None and not math.isfinite(
            self.reaction_force
        ):
            raise ValueError("reaction_force must be finite when available")
        if self.displacement is None:
            if self.converged:
                raise ValueError("a converged FEAResult requires displacement")
            return
        displacement = np.array(self.displacement, dtype=float, copy=True)
        if displacement.shape != self.mesh.coordinates.shape:
            raise ValueError("displacement must have shape (N, 2)")
        if not np.all(np.isfinite(displacement)):
            raise ValueError("displacement must contain finite values")
        displacement.setflags(write=False)
        object.__setattr__(self, "displacement", displacement)

    @property
    def deformed_mesh(self) -> Any:
        """Return the converged pad mesh view without rebuilding topology."""
        if not self.converged or self.displacement is None:
            raise RuntimeError(
                "a deformed mesh is unavailable because the solve did not converge"
            )
        return self.mesh.deformed(
            self.displacement,
            metadata={"condition": "loaded"},
        )


def solve(
    tip: Fingertip,
    mesh: FingertipMesh,
    *,
    indentation: float,
    steps: int = 48,
    indenter: IndenterSettings | None = None,
    internal_contact: str = "three_pairs",
) -> FEAResult:
    """Solve one indentation case and return only solver-independent data."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    if not isinstance(mesh, FingertipMesh):
        raise TypeError("mesh must be the FingertipMesh returned by tip.mesh()")
    if mesh.parameters != tip.parameters:
        raise ValueError("mesh parameters do not match the fingertip")

    final_displacements: dict[int, tuple[float, float]] | None = None

    def capture(step: ConvergedIndentationStep) -> None:
        nonlocal final_displacements
        final_displacements = {
            int(node_id): (float(value[0]), float(value[1]))
            for node_id, value in step.displacements.items()
        }

    details, _ = run_indentation_case(
        tip.geometry,
        mesh.settings.level,
        IndentationSettings(
            indentation_mm=indentation,
            number_of_steps=steps,
        ),
        indenter_settings=indenter,
        internal_contact_configuration=internal_contact,
        mesh_override=mesh,
        converged_step_observer=capture,
    )
    pad_mesh = mesh.pad
    displacement = (
        None
        if final_displacements is None
        else np.asarray(
            [final_displacements[int(node_id)] for node_id in pad_mesh.node_ids],
            dtype=float,
        )
    )
    final = details.get("final", {})
    reaction = final.get("indenter_normal_reaction_n")
    converged = details.get("solve_status") == "PASS" and displacement is not None
    return FEAResult(
        mesh=pad_mesh,
        displacement=displacement,
        reaction_force=None if reaction is None else float(reaction),
        contact=dict(final.get("contact_groups", {})),
        converged=converged,
        details=details,
    )


__all__ = ["FEAResult", "solve"]
