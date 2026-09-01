"""Intel RealSense color-frame acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ColorFrame:
    """One owned RGB camera frame with device timing metadata."""

    rgb: np.ndarray
    timestamp_ms: float
    frame_number: int

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be an H x W x 3 uint8 array")
        if not np.isfinite(self.timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        if self.frame_number < 0:
            raise ValueError("frame_number must be nonnegative")
        rgb = rgb.copy()
        rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)


class RealSenseColorCamera:
    """Own one pyrealsense2 color pipeline and its explicit lifecycle."""

    def __init__(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        serial_number: str | None = None,
    ) -> None:
        for name, value in (("width", width), ("height", height), ("fps", fps)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if serial_number is not None and not serial_number.strip():
            raise ValueError("serial_number must be nonempty when supplied")

        self.width = width
        self.height = height
        self.fps = fps
        self.requested_serial_number = serial_number
        self.device_name: str | None = None
        self.serial_number: str | None = None
        self._rs: Any | None = None
        self._pipeline: Any | None = None
        self._color_sensor: Any | None = None

    @property
    def is_running(self) -> bool:
        """Whether this object currently owns a running RealSense pipeline."""

        return self._pipeline is not None

    def start(self) -> None:
        """Open the selected camera and start its RGB color stream."""

        if self.is_running:
            raise RuntimeError("RealSense camera is already running")
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "pyrealsense2 is required for RealSenseColorCamera"
            ) from error

        pipeline = rs.pipeline()
        config = rs.config()
        if self.requested_serial_number is not None:
            config.enable_device(self.requested_serial_number)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )

        try:
            profile = pipeline.start(config)
        except Exception as error:
            raise RuntimeError(f"failed to start RealSense color stream: {error}") from error

        device = profile.get_device()
        try:
            color_sensor = device.first_color_sensor()
        except Exception as error:
            pipeline.stop()
            raise RuntimeError(
                f"RealSense device did not expose a color sensor: {error}"
            ) from error
        self.device_name = str(device.get_info(rs.camera_info.name))
        self.serial_number = str(device.get_info(rs.camera_info.serial_number))
        self._rs = rs
        self._pipeline = pipeline
        self._color_sensor = color_sensor

    def lock_color_photometric_controls(self) -> dict[str, float | None]:
        """Freeze the current RGB exposure, gain, and white balance.

        Automatic controls must be allowed to settle before this method is
        called. The returned values are read back after locking. A ``None``
        value means that the color sensor did not expose that optional control.
        """

        if (
            self._pipeline is None
            or self._rs is None
            or self._color_sensor is None
        ):
            raise RuntimeError("RealSense camera is not running")

        rs = self._rs
        sensor = self._color_sensor
        auto_exposure = rs.option.enable_auto_exposure
        exposure = rs.option.exposure
        gain = rs.option.gain
        auto_white_balance = rs.option.enable_auto_white_balance
        white_balance = rs.option.white_balance

        if not sensor.supports(auto_exposure) or not sensor.supports(exposure):
            raise RuntimeError(
                "RealSense color sensor cannot freeze exposure: "
                "enable_auto_exposure and exposure controls are required"
            )

        retained_exposure = float(sensor.get_option(exposure))
        retained_gain = float(sensor.get_option(gain)) if sensor.supports(gain) else None
        can_lock_white_balance = sensor.supports(
            auto_white_balance
        ) and sensor.supports(white_balance)
        retained_white_balance = (
            float(sensor.get_option(white_balance))
            if can_lock_white_balance
            else None
        )

        try:
            sensor.set_option(auto_exposure, 0.0)
            sensor.set_option(exposure, retained_exposure)
            if retained_gain is not None:
                sensor.set_option(gain, retained_gain)
            if retained_white_balance is not None:
                sensor.set_option(auto_white_balance, 0.0)
                sensor.set_option(white_balance, retained_white_balance)
        except Exception as error:
            raise RuntimeError(
                f"failed to lock RealSense color photometric controls: {error}"
            ) from error

        if sensor.get_option(auto_exposure) != 0.0:
            raise RuntimeError("RealSense color auto exposure remained enabled")
        if (
            retained_white_balance is not None
            and sensor.get_option(auto_white_balance) != 0.0
        ):
            raise RuntimeError("RealSense color auto white balance remained enabled")

        return {
            "exposure": float(sensor.get_option(exposure)),
            "gain": (
                float(sensor.get_option(gain)) if retained_gain is not None else None
            ),
            "white_balance": (
                float(sensor.get_option(white_balance))
                if retained_white_balance is not None
                else None
            ),
        }

    def read(self, *, timeout_ms: int = 5000) -> ColorFrame:
        """Block for the next color frame and return an owned RGB copy."""

        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        if self._pipeline is None:
            raise RuntimeError("RealSense camera is not running")

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
            color_frame = frames.get_color_frame()
        except Exception as error:
            raise RuntimeError(f"failed to read RealSense color frame: {error}") from error
        if not color_frame:
            raise RuntimeError("RealSense frameset did not contain a color frame")

        bgr = np.asanyarray(color_frame.get_data())
        if bgr.shape != (self.height, self.width, 3) or bgr.dtype != np.uint8:
            raise RuntimeError(
                "RealSense returned an unexpected color frame: "
                f"shape={bgr.shape}, dtype={bgr.dtype}"
            )
        rgb = bgr[:, :, ::-1]
        return ColorFrame(
            rgb=rgb,
            timestamp_ms=float(color_frame.get_timestamp()),
            frame_number=int(color_frame.get_frame_number()),
        )

    def stop(self) -> None:
        """Stop streaming. Calling this on a stopped camera is harmless."""

        pipeline = self._pipeline
        self._pipeline = None
        self._color_sensor = None
        self._rs = None
        if pipeline is not None:
            pipeline.stop()

    def __enter__(self) -> RealSenseColorCamera:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


__all__ = ["ColorFrame", "RealSenseColorCamera"]
