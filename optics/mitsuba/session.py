"""Persistent in-memory Mitsuba render session for one fixed topology."""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np

from model.fingertip_sensor_model import FingertipSensorModel
from optics.geometry.deformation_state import PadDeformationState2D
from optics.geometry.extrusion import ExtrudedOpticalMeshTemplate
from optics.geometry.pad_mesh_template import PadMeshTemplate2D
from optics.mitsuba.parameters import MitsubaCameraParameters, MitsubaRenderSettings
from optics.mitsuba.result import CameraRenderResult
from optics.mitsuba.scene import build_in_memory_mitsuba_scene


class MitsubaSessionError(RuntimeError):
    """Raised when a persistent Mitsuba session cannot satisfy its contract."""


class MitsubaRenderSession:
    """Reusable, non-thread-safe Mitsuba scene for one fixed mesh topology."""

    def __init__(
        self,
        *,
        sensor_model: FingertipSensorModel,
        mesh_template: PadMeshTemplate2D,
        extrusion: ExtrudedOpticalMeshTemplate,
        camera: MitsubaCameraParameters,
        settings: MitsubaRenderSettings | None = None,
    ) -> None:
        self.sensor_model = sensor_model
        self.mesh_template = mesh_template
        self.extrusion = extrusion
        self.camera = camera
        self.settings = settings or MitsubaRenderSettings()
        if extrusion.node_count_2d != len(mesh_template.node_ids):
            raise MitsubaSessionError(
                "extrusion and mesh template node counts differ"
            )
        try:
            import mitsuba as mi
        except ImportError as exc:
            raise MitsubaSessionError(
                "Mitsuba is required for camera rendering; install the "
                "optional 'optics' dependency"
            ) from exc
        active_variant = mi.variant()
        if active_variant is None:
            mi.set_variant(self.settings.variant)
        elif active_variant != self.settings.variant:
            raise MitsubaSessionError(
                f"Mitsuba variant '{active_variant}' is active; this session "
                f"requires '{self.settings.variant}'"
            )
        self._mi = mi
        zero_state = PadDeformationState2D.zero(mesh_template)
        reference_vertices = extrusion.vertices_for_state(
            mesh_template,
            zero_state,
        )
        self._scene = build_in_memory_mitsuba_scene(
            mi,
            sensor_model=sensor_model,
            extrusion=extrusion,
            vertices_mm=reference_vertices,
            camera=camera,
            settings=self.settings,
        )
        self._scene_parameters = mi.traverse(self._scene)
        available_keys = tuple(str(key) for key in self._scene_parameters.keys())
        self._vertex_position_key = self._required_key(
            "pad.vertex_positions",
            available_keys,
        )
        self._led_intensity_key = self._required_key(
            "led.intensity.value",
            available_keys,
        )
        self._current_state_metadata: dict[str, Any] = dict(zero_state.metadata)
        self._relative_led_power = sensor_model.led.relative_radiant_power

    @staticmethod
    def _required_key(expected: str, available_keys: tuple[str, ...]) -> str:
        if expected not in available_keys:
            available = ", ".join(available_keys) or "<none>"
            raise MitsubaSessionError(
                f"required Mitsuba scene parameter '{expected}' is absent; "
                f"available keys: {available}"
            )
        return expected

    def update_state(self, state: PadDeformationState2D) -> None:
        """Update only pad vertex positions for one new deformation state."""
        vertices = self.extrusion.vertices_for_state(
            self.mesh_template,
            state,
        )
        self._scene_parameters[self._vertex_position_key] = np.asarray(
            vertices,
            dtype=np.float32,
        ).reshape(-1)
        self._scene_parameters.update()
        self._current_state_metadata = dict(state.metadata)

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
            * np.asarray(self.sensor_model.led.emission_rgb, dtype=float)
        )
        self._scene_parameters[self._led_intensity_key] = intensity.tolist()
        self._scene_parameters.update()
        self._relative_led_power = float(relative_radiant_power)

    def render(self, *, spp: int | None = None) -> CameraRenderResult:
        """Render the current in-memory state as raw linear RGB."""
        sample_count = self.settings.spp if spp is None else spp
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 1
        ):
            raise ValueError("spp must be a positive integer")
        rendered = np.asarray(
            self._mi.render(self._scene, spp=sample_count),
            dtype=float,
        )
        if rendered.ndim != 3 or rendered.shape[2] < 3:
            raise MitsubaSessionError(
                "Mitsuba returned an image without three RGB channels"
            )
        return CameraRenderResult(
            linear_rgb=rendered[:, :, :3],
            spp=sample_count,
            relative_led_power=self._relative_led_power,
            state_metadata=self._current_state_metadata,
        )

    def render_state(
        self,
        state: PadDeformationState2D,
        *,
        spp: int | None = None,
        relative_led_power: float | None = None,
    ) -> CameraRenderResult:
        """Update deformation and optional source power, then render."""
        self.update_state(state)
        if relative_led_power is not None:
            self.set_led_relative_power(relative_led_power)
        return self.render(spp=spp)
