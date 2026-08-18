"""Public neutral facade for fingertip indentation solves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

from fem.indentation import (
    ConvergedIndentationStep,
    IndentationSettings,
    run_indentation_case,
)
from mesh.indenter import (
    IndenterPose2D,
    IndenterSettings,
    build_normal_indenter_fixture_at_x,
    pose_from_fixture,
)
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
    indenter_pose: IndenterPose2D | None = None
    reference_mesh: FingertipMesh | None = None
    element_von_mises_stress_mpa: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        if self.reaction_force is not None and not math.isfinite(
            self.reaction_force
        ):
            raise ValueError("reaction_force must be finite when available")
        if not isinstance(self.mesh, PadMesh):
            raise TypeError("FEAResult.mesh must be a PadMesh")
        if self.reference_mesh is not None:
            if not isinstance(self.reference_mesh, FingertipMesh):
                raise TypeError("reference_mesh must be a FingertipMesh when supplied")
            if not np.array_equal(self.reference_mesh.pad.node_ids, self.mesh.node_ids):
                raise ValueError("reference_mesh pad topology does not match mesh")
            if not np.array_equal(
                self.reference_mesh.pad.triangles,
                self.mesh.triangles,
            ):
                raise ValueError("reference_mesh pad elements do not match mesh")
        if self.element_von_mises_stress_mpa is not None:
            stress = {
                int(element_id): float(value)
                for element_id, value in self.element_von_mises_stress_mpa.items()
            }
            if any(
                not math.isfinite(value) or value < 0.0
                for value in stress.values()
            ):
                raise ValueError(
                    "element_von_mises_stress_mpa must be finite and nonnegative"
                )
            object.__setattr__(
                self,
                "element_von_mises_stress_mpa",
                MappingProxyType(stress),
            )
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
    surface_x_mm: float = 0.0,
    steps: int = 48,
    indenter: IndenterSettings | None = None,
    internal_contact: str = "sides_separate",
    basal_interface: str = "bonded",
) -> FEAResult:
    """Solve one local-normal indentation case and return neutral data."""
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

    fixture = build_normal_indenter_fixture_at_x(
        tip.geometry,
        surface_x_mm,
        indenter,
    )
    details, _ = run_indentation_case(
        tip.geometry,
        mesh.settings.level,
        IndentationSettings(
            indentation_mm=indentation,
            number_of_steps=steps,
        ),
        internal_contact_configuration=internal_contact,
        basal_interface=basal_interface,
        mesh_override=mesh,
        fixture_override=fixture,
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
    stress = final.get("element_von_mises_stress_mpa")
    indenter_pose = None
    if converged:
        external = final.get("contact_groups", {}).get(
            "external_pad_indenter", {}
        )
        active_node_ids = tuple(
            int(node_id)
            for node_id in external.get("active_slave_node_ids", [])
        )
        active = set(active_node_ids)
        deformed_coordinates = pad_mesh.coordinates + displacement
        patch_segments = [
            LineString(
                [
                    deformed_coordinates[int(first)],
                    deformed_coordinates[int(second)],
                ]
            )
            for first, second in pad_mesh.boundaries.get("pad_outer_arc", ())
            if int(pad_mesh.node_ids[int(first)]) in active
            and int(pad_mesh.node_ids[int(second)]) in active
        ]
        contact_patch = None
        if patch_segments:
            merged = unary_union(patch_segments)
            if isinstance(merged, LineString | MultiLineString):
                contact_patch = merged
        indenter_pose = pose_from_fixture(
            fixture,
            float(final["prescribed_indenter_travel_mm"]),
            contact_patch=contact_patch,
            active_contact_node_ids=active_node_ids,
        )
    return FEAResult(
        mesh=pad_mesh,
        displacement=displacement,
        reaction_force=None if reaction is None else float(reaction),
        contact=dict(final.get("contact_groups", {})),
        converged=converged,
        details=details,
        indenter_pose=indenter_pose,
        reference_mesh=mesh,
        element_von_mises_stress_mpa=stress,
    )


__all__ = ["FEAResult", "solve"]
