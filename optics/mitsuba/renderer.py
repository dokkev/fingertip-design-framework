"""Small public facade for the optional Mitsuba validator."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from model import Fingertip
from optics.geometry.extrusion import _ExtrudedMesh
from optics.mitsuba.parameters import (
    Camera,
    RenderSettings,
    _default_camera,
)
from optics.mitsuba.result import RenderResult
from optics.mitsuba.session import MitsubaError, _MitsubaSession


def _pad_view(mesh: Any) -> Any:
    if hasattr(mesh, "coordinates") and hasattr(mesh, "triangles"):
        return mesh
    try:
        return mesh.pad
    except AttributeError as exc:
        raise TypeError(
            "mesh must be a FingertipMesh, PadMesh, or deformed pad mesh"
        ) from exc


class MitsubaRenderer:
    """Render many states of one fingertip and one fixed mesh topology."""

    def __init__(
        self,
        tip: Fingertip,
        mesh: Any,
        *,
        depth_mm: float | None = None,
        camera: Camera | None = None,
        settings: RenderSettings | None = None,
    ) -> None:
        if not isinstance(tip, Fingertip):
            raise TypeError("tip must be a Fingertip")
        selected_settings = settings or RenderSettings()
        if depth_mm is not None:
            selected_settings = replace(
                selected_settings,
                optical_depth_mm=depth_mm,
            )
        supplied_mesh = _pad_view(mesh)
        reference_mesh = supplied_mesh.reference_mesh
        extrusion = _ExtrudedMesh.from_pad_mesh(
            reference_mesh,
            depth_mm=selected_settings.optical_depth_mm,
        )
        self._reference_mesh = reference_mesh
        self._session = _MitsubaSession(
            tip=tip,
            reference_mesh=reference_mesh,
            extrusion=extrusion,
            camera=camera or _default_camera(tip, reference_mesh),
            settings=selected_settings,
        )

    def _checked_view(self, mesh: Any) -> Any:
        view = _pad_view(mesh)
        reference = view.reference_mesh
        if not (
            np.array_equal(reference.node_ids, self._reference_mesh.node_ids)
            and np.array_equal(reference.triangles, self._reference_mesh.triangles)
            and np.array_equal(
                reference.boundary_edges,
                self._reference_mesh.boundary_edges,
            )
            and np.array_equal(
                reference.coordinates,
                self._reference_mesh.coordinates,
            )
        ):
            raise MitsubaError(
                "rendered mesh must use the renderer's fixed reference topology"
            )
        return view

    def render(
        self,
        mesh: Any | None = None,
        *,
        displacement: np.ndarray | None = None,
        spp: int | None = None,
        relative_led_power: float | None = None,
    ) -> RenderResult:
        """Render a mesh view or a displacement on the reference mesh."""
        if mesh is not None and displacement is not None:
            raise ValueError("pass either mesh or displacement, not both")
        if displacement is not None:
            view = self._reference_mesh.deformed(displacement)
        elif mesh is not None:
            view = self._checked_view(mesh)
        else:
            view = self._reference_mesh
        return self._session.render_mesh(
            view,
            spp=spp,
            relative_led_power=relative_led_power,
        )


__all__ = [
    "Camera",
    "MitsubaRenderer",
    "RenderResult",
    "RenderSettings",
]
