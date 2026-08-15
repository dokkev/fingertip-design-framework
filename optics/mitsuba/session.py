"""Persistent in-memory Mitsuba render session for one fixed topology."""

from __future__ import annotations

from math import isfinite
from typing import Any, Sequence

import numpy as np

from mesh import PadMesh
from model.fingertip import Fingertip
from optics.geometry.extrusion import _ExtrudedMesh
from optics.mitsuba.parameters import Camera, RenderSettings
from optics.mitsuba.result import RenderResult
from optics.mitsuba.scene import build_in_memory_mitsuba_scene


class MitsubaError(RuntimeError):
    """Raised when a persistent Mitsuba session cannot satisfy its contract."""


class _MitsubaSession:
    """Reusable, non-thread-safe Mitsuba scene for one fixed mesh topology."""

    def __init__(
        self,
        *,
        tip: Fingertip,
        reference_mesh: PadMesh,
        extrusion: _ExtrudedMesh,
        camera: Camera,
        settings: RenderSettings | None = None,
        source_positions_mm: Sequence[tuple[float, float, float]] | None = None,
    ) -> None:
        self.tip = tip
        self.reference_mesh = reference_mesh
        self.extrusion = extrusion
        self.camera = camera
        self.settings = settings or RenderSettings()
        self.source_positions_mm = (
            None if source_positions_mm is None else tuple(source_positions_mm)
        )
        if extrusion.node_count_2d != len(reference_mesh.node_ids):
            raise MitsubaError(
                "extrusion and reference mesh node counts differ"
            )
        try:
            import mitsuba as mi
        except ImportError as exc:
            raise MitsubaError(
                "Mitsuba is required for camera rendering; install the "
                "optional 'optics' dependency"
            ) from exc
        active_variant = mi.variant()
        if active_variant is None:
            mi.set_variant(self.settings.variant)
        elif active_variant != self.settings.variant:
            raise MitsubaError(
                f"Mitsuba variant '{active_variant}' is active; this session "
                f"requires '{self.settings.variant}'"
            )
        self._mi = mi
        reference_vertices = extrusion.vertices_for_mesh(reference_mesh)
        self._scene = build_in_memory_mitsuba_scene(
            mi,
            tip=tip,
            extrusion=extrusion,
            vertices_mm=reference_vertices,
            camera=camera,
            settings=self.settings,
            source_positions_mm=self.source_positions_mm,
        )
        self._scene_parameters = mi.traverse(self._scene)
        available_keys = tuple(str(key) for key in self._scene_parameters.keys())
        self._vertex_position_key = self._required_key(
            "pad.vertex_positions",
            available_keys,
        )
        if self.source_positions_mm is None or len(self.source_positions_mm) == 1:
            source_names = ("led",)
        else:
            source_names = tuple(
                f"led_{index}" for index in range(len(self.source_positions_mm))
            )
        self._led_intensity_keys = tuple(
            self._required_key(f"{name}.intensity.value", available_keys)
            for name in source_names
        )
        self._current_state_metadata: dict[str, Any] = dict(
            reference_mesh.metadata
        )
        self._relative_led_power = tip.led.relative_radiant_power

    @staticmethod
    def _required_key(expected: str, available_keys: tuple[str, ...]) -> str:
        if expected not in available_keys:
            available = ", ".join(available_keys) or "<none>"
            raise MitsubaError(
                f"required Mitsuba scene parameter '{expected}' is absent; "
                f"available keys: {available}"
            )
        return expected

    def update_mesh(self, mesh: Any) -> None:
        """Update only pad vertex positions for one mesh state."""
        vertices = self.extrusion.vertices_for_mesh(mesh)
        self._scene_parameters[self._vertex_position_key] = np.asarray(
            vertices,
            dtype=np.float32,
        ).reshape(-1)
        self._scene_parameters.update()
        self._current_state_metadata = dict(mesh.metadata)

    def set_led_relative_power(self, relative_radiant_power: float) -> None:
        """Update point-emitter intensity without rebuilding mesh geometry."""
        if (
            not isfinite(relative_radiant_power)
            or relative_radiant_power < 0.0
        ):
            raise ValueError(
                "relative_radiant_power must be finite and nonnegative"
            )
        intensity = (
            self.settings.point_emitter_scale
            * relative_radiant_power
            * np.asarray(self.tip.led.emission_rgb, dtype=float)
        )
        for key in self._led_intensity_keys:
            self._scene_parameters[key] = intensity.tolist()
        self._scene_parameters.update()
        self._relative_led_power = float(relative_radiant_power)

    def render(
        self,
        *,
        spp: int | None = None,
        seed: int | None = None,
    ) -> RenderResult:
        """Render the current in-memory state as raw linear RGB."""
        sample_count = self.settings.spp if spp is None else spp
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 1
        ):
            raise ValueError("spp must be a positive integer")
        render_kwargs: dict[str, Any] = {"spp": sample_count}
        if seed is not None:
            render_kwargs["seed"] = seed
        rendered = np.asarray(self._mi.render(self._scene, **render_kwargs), dtype=float)
        if rendered.ndim != 3 or rendered.shape[2] < 3:
            raise MitsubaError(
                "Mitsuba returned an image without three RGB channels"
            )
        return RenderResult(
            linear_rgb=rendered[:, :, :3],
            spp=sample_count,
            relative_led_power=self._relative_led_power,
            state_metadata=self._current_state_metadata,
        )

    def render_mesh(
        self,
        mesh: Any,
        *,
        spp: int | None = None,
        relative_led_power: float | None = None,
        seed: int | None = None,
    ) -> RenderResult:
        """Update mesh coordinates and optional source power, then render."""
        self.update_mesh(mesh)
        if relative_led_power is not None:
            self.set_led_relative_power(relative_led_power)
        return self.render(spp=spp, seed=seed)
