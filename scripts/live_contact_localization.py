"""Run live five-LED contact localization from a RealSense D435 color stream."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.hardware import RealSenseColorCamera  # noqa: E402
from experiments.localization import (  # noqa: E402
    LedArrayGeometry,
    brightest_red_features,
    contact_image_point,
    detect_led_array,
    estimate_contact_position,
    track_led_array,
)
from lumo.fingertip import LED_CENTERS_Y_MM  # noqa: E402


CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CAMERA_SERIAL_NUMBER: str | None = None
CALIBRATION_FRAME_COUNT = 30
# Physical positions corresponding to landmarks ordered top-to-bottom in the
# displayed camera image. Reverse this tuple if the camera is mounted opposite.
LED_POSITIONS_IN_IMAGE_ORDER_MM = np.asarray(LED_CENTERS_Y_MM, dtype=np.float64)
WINDOW_NAME = "LUMO live contact localization"


def _draw_geometry(image: np.ndarray, geometry: LedArrayGeometry) -> None:
    for index, (landmark, polygon) in enumerate(
        zip(
            geometry.landmarks_xy_px,
            geometry.roi_polygons_xy_px,
            strict=True,
        ),
        start=1,
    ):
        cv2.polylines(
            image,
            [np.rint(polygon).astype(np.int32)],
            True,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        center = tuple(np.rint(landmark).astype(int))
        cv2.circle(image, center, 4, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"LED {index}",
            (center[0] + 6, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_contact_point(image: np.ndarray, point_xy_px: np.ndarray) -> None:
    center = tuple(np.rint(point_xy_px).astype(int))
    cv2.circle(image, center, 10, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(image, center, 7, (0, 70, 255), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        "contact",
        (center[0] + 12, center[1] - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 70, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_text(image: np.ndarray, lines: list[str]) -> None:
    box_height = 18 + 22 * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (430, box_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 32 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def _draw_response_panel(
    image: np.ndarray,
    response: np.ndarray | None,
) -> np.ndarray:
    panel_width = 250
    panel = np.full((image.shape[0], panel_width, 3), 26, dtype=np.uint8)
    cv2.putText(
        panel,
        "Baseline-relative red response",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    if response is None:
        cv2.putText(
            panel,
            "Press b while unloaded",
            (14, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        return np.hstack((image, panel))

    scale = max(float(np.max(np.abs(response))), 1.0)
    center_x = 112
    for index, value in enumerate(response):
        y = 82 + 66 * index
        length = round(88.0 * abs(float(value)) / scale)
        color = (50, 210, 80) if value >= 0.0 else (60, 90, 230)
        endpoint = center_x + length if value >= 0.0 else center_x - length
        cv2.line(panel, (center_x, y), (endpoint, y), color, 12, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"LED {index + 1}",
            (14, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{value:+.1f}",
            (180, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.line(panel, (center_x, 54), (center_x, 372), (115, 115, 115), 1)
    return np.hstack((image, panel))


def main() -> None:
    calibration_frames: deque[np.ndarray] = deque(maxlen=CALIBRATION_FRAME_COUNT)
    geometry: LedArrayGeometry | None = None
    baseline: np.ndarray | None = None
    calibration_error: str | None = None
    previous_rgb: np.ndarray | None = None
    previous_wall_time = perf_counter()
    displayed_fps = 0.0

    camera = RealSenseColorCamera(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        serial_number=CAMERA_SERIAL_NUMBER,
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        with camera:
            print(
                f"camera: {camera.device_name}, serial={camera.serial_number}, "
                f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}"
            )
            print("keys: b=set unloaded baseline, r=recalibrate LEDs, q/esc=quit")
            while True:
                frame = camera.read()
                rgb = frame.rgb
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                features: np.ndarray | None = None
                response: np.ndarray | None = None
                lines = ["q/esc quit | r recalibrate | b unloaded baseline"]

                if geometry is not None and previous_rgb is not None:
                    try:
                        geometry = track_led_array(previous_rgb, rgb, geometry)
                    except RuntimeError as error:
                        print(f"LED tracking lost: {error}; automatically recalibrating")
                        geometry = None
                        baseline = None
                        calibration_error = None
                        calibration_frames.clear()
                        lines.append("Camera pose changed: automatically recalibrating LEDs")

                if geometry is None:
                    if calibration_error is None:
                        calibration_frames.append(rgb)
                        count = len(calibration_frames)
                        lines.append(
                            f"LED calibration: {count}/{CALIBRATION_FRAME_COUNT} - hold camera still"
                        )
                        if count == CALIBRATION_FRAME_COUNT:
                            try:
                                geometry = detect_led_array(np.stack(calibration_frames))
                            except RuntimeError as error:
                                calibration_error = str(error)
                                print(f"LED calibration failed: {calibration_error}")
                            calibration_frames.clear()
                            if geometry is not None:
                                print(
                                    "LED image landmarks [x,y] px: "
                                    f"{np.round(geometry.landmarks_xy_px, 2).tolist()}"
                                )
                    else:
                        lines.extend(
                            (
                                f"LED calibration failed: {calibration_error}",
                                "Reframe the five visible LEDs, then press r",
                            )
                        )
                else:
                    _draw_geometry(display, geometry)
                    lines.append("LED image positions: tracking")
                    features = brightest_red_features(rgb, geometry)
                    if baseline is None:
                        lines.append("Press b with the fingertip unloaded")
                    else:
                        estimate = estimate_contact_position(
                            features,
                            baseline,
                            LED_POSITIONS_IN_IMAGE_ORDER_MM,
                        )
                        response = estimate.response
                        marker = contact_image_point(estimate, geometry)
                        if marker is not None:
                            _draw_contact_point(display, marker)
                        position_text = (
                            "unavailable"
                            if estimate.position_mm is None
                            else f"{estimate.position_mm:+.2f} mm"
                        )
                        lines.extend(
                            (
                                f"Contact position: {position_text}",
                                f"Peak LED: {estimate.predicted_led_index + 1} | "
                                f"top-2 margin: {estimate.top_two_margin:.2f} DN",
                            )
                        )

                now = perf_counter()
                instantaneous_fps = 1.0 / max(now - previous_wall_time, 1.0e-9)
                previous_wall_time = now
                displayed_fps = 0.90 * displayed_fps + 0.10 * instantaneous_fps
                lines.append(f"Frame {frame.frame_number} | {displayed_fps:.1f} FPS")
                _draw_text(display, lines)
                visualization = _draw_response_panel(display, response)
                cv2.imshow(WINDOW_NAME, visualization)

                key = cv2.waitKey(1) & 0xFF
                try:
                    window_visible = cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    )
                except cv2.error:
                    window_visible = 0.0
                if key in (27, ord("q")) or window_visible < 1.0:
                    break
                if key == ord("r"):
                    geometry = None
                    baseline = None
                    calibration_error = None
                    calibration_frames.clear()
                    print("recalibrating LED geometry")
                elif key == ord("b") and features is not None:
                    baseline = features.copy()
                    print(f"unloaded baseline set: {np.round(baseline, 3).tolist()}")
                previous_rgb = rgb
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
