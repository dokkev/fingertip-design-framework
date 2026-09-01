"""Run live five-LED contact localization from a RealSense D435 color stream."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from time import perf_counter, sleep

import cv2
import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.hardware import ColorFrame, RealSenseColorCamera  # noqa: E402
from experiments.localization import (  # noqa: E402
    CONTACT_Z_THRESHOLD,
    FEATURE_NOISE_FLOOR_DN,
    LedArrayGeometry,
    brightest_red_features,
    contact_image_point,
    detect_led_array,
    estimate_contact_position,
    reanchor_led_array,
    track_led_array,
    unloaded_baseline_statistics,
)
from lumo.fingertip import LED_CENTERS_Y_MM  # noqa: E402


CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CAMERA_SERIAL_NUMBER: str | None = None
PHOTOMETRIC_WARMUP_FRAME_COUNT = 30
CALIBRATION_FRAME_COUNT = 30
BASELINE_FRAME_COUNT = 30
FEATURE_MEDIAN_WINDOW = 3
NO_CONTACT_REANCHOR_FRAME_COUNT = 30
RESPONSE_PANEL_MAX_Z = 8.0
CAMERA_FRAME_TIMEOUT_MS = 2000
CAMERA_RECONNECT_ATTEMPTS = 10
CAMERA_RECONNECT_DELAY_S = 1.0
# Physical positions corresponding to landmarks ordered top-to-bottom in the
# displayed camera image. Reverse this tuple if the camera is mounted opposite.
LED_POSITIONS_IN_IMAGE_ORDER_MM = np.asarray(LED_CENTERS_Y_MM, dtype=np.float64)
WINDOW_NAME = "LUMO live contact localization"


def _warm_up_and_lock_photometric_controls(
    camera: RealSenseColorCamera,
) -> dict[str, float | None]:
    print(
        "settling automatic camera controls: "
        f"{PHOTOMETRIC_WARMUP_FRAME_COUNT} frames"
    )
    for _ in range(PHOTOMETRIC_WARMUP_FRAME_COUNT):
        camera.read(timeout_ms=CAMERA_FRAME_TIMEOUT_MS)
    controls = camera.lock_color_photometric_controls()
    print("photometric controls locked:")
    for name in ("exposure", "gain", "white_balance"):
        value = controls[name]
        print(f"  {name} = {'unsupported' if value is None else value}")
    return controls


def _read_with_reconnect(
    camera: RealSenseColorCamera,
) -> tuple[ColorFrame, bool]:
    try:
        return camera.read(timeout_ms=CAMERA_FRAME_TIMEOUT_MS), False
    except RuntimeError as error:
        last_error = error
        camera.stop()

    for attempt in range(1, CAMERA_RECONNECT_ATTEMPTS + 1):
        print(
            "RealSense stream lost; reconnecting "
            f"{attempt}/{CAMERA_RECONNECT_ATTEMPTS}: {last_error}"
        )
        sleep(CAMERA_RECONNECT_DELAY_S)
        try:
            camera.start()
            _warm_up_and_lock_photometric_controls(camera)
            return camera.read(timeout_ms=CAMERA_FRAME_TIMEOUT_MS), True
        except RuntimeError as error:
            last_error = error
            camera.stop()
    raise RuntimeError(
        "RealSense reconnect budget exhausted after "
        f"{CAMERA_RECONNECT_ATTEMPTS} attempts: {last_error}"
    ) from last_error


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
    unloaded_noise_sigma: np.ndarray | None,
) -> np.ndarray:
    panel_width = 340
    panel = np.full((image.shape[0], panel_width, 3), 26, dtype=np.uint8)
    cv2.putText(
        panel,
        "Red response: fixed unloaded-noise scale",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    if response is None or unloaded_noise_sigma is None:
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

    noise_sigma = np.asarray(unloaded_noise_sigma, dtype=np.float64)
    if noise_sigma.shape != (5,) or not np.all(np.isfinite(noise_sigma)):
        raise ValueError("unloaded_noise_sigma must be a finite length-five vector")
    standardized = np.asarray(response, dtype=np.float64) / np.maximum(
        noise_sigma,
        FEATURE_NOISE_FLOOR_DN,
    )
    zero_x = 165
    scale_width = 95
    for tick_z in (0.0, 2.0, CONTACT_Z_THRESHOLD, RESPONSE_PANEL_MAX_Z):
        x = zero_x + round(scale_width * tick_z / RESPONSE_PANEL_MAX_Z)
        color = (
            (0, 150, 255)
            if tick_z == CONTACT_Z_THRESHOLD
            else (90, 90, 90)
        )
        cv2.line(panel, (x, 54), (x, 372), color, 1, cv2.LINE_AA)
        if tick_z == 0.0:
            label = "0"
        elif tick_z == CONTACT_Z_THRESHOLD:
            label = "4 gate"
        elif tick_z == RESPONSE_PANEL_MAX_Z:
            label = "8+"
        else:
            label = f"{tick_z:g}"
        cv2.putText(
            panel,
            label,
            (x - 7, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (185, 185, 185),
            1,
            cv2.LINE_AA,
        )

    for index, (value, z_score) in enumerate(
        zip(response, standardized, strict=True)
    ):
        y = 82 + 66 * index
        clipped_z = float(
            np.clip(z_score, -RESPONSE_PANEL_MAX_Z, RESPONSE_PANEL_MAX_Z)
        )
        length = round(scale_width * abs(clipped_z) / RESPONSE_PANEL_MAX_Z)
        color = (50, 210, 80) if value >= 0.0 else (60, 90, 230)
        endpoint = zero_x + length if clipped_z >= 0.0 else zero_x - length
        cv2.line(panel, (zero_x, y), (endpoint, y), color, 12, cv2.LINE_AA)
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
            (278, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
    return np.hstack((image, panel))


def main() -> None:
    calibration_frames: deque[np.ndarray] = deque(maxlen=CALIBRATION_FRAME_COUNT)
    baseline_samples: deque[np.ndarray] = deque(maxlen=BASELINE_FRAME_COUNT)
    feature_history: deque[np.ndarray] = deque(maxlen=FEATURE_MEDIAN_WINDOW)
    geometry: LedArrayGeometry | None = None
    baseline: np.ndarray | None = None
    unloaded_noise_sigma: np.ndarray | None = None
    baseline_collecting = False
    no_contact_frame_count = 0
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
            _warm_up_and_lock_photometric_controls(camera)
            print("keys: b=set unloaded baseline, r=recalibrate LEDs, q/esc=quit")
            while True:
                frame, reconnected = _read_with_reconnect(camera)
                if reconnected:
                    print(
                        f"camera reconnected: {camera.device_name}, "
                        f"serial={camera.serial_number}"
                    )
                    geometry = None
                    baseline = None
                    unloaded_noise_sigma = None
                    baseline_collecting = False
                    no_contact_frame_count = 0
                    calibration_error = None
                    previous_rgb = None
                    calibration_frames.clear()
                    baseline_samples.clear()
                    feature_history.clear()
                rgb = frame.rgb
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                features: np.ndarray | None = None
                response: np.ndarray | None = None
                lines = ["q/esc quit | r recalibrate | b unloaded baseline"]
                lines.append("Photometric controls: locked")

                if geometry is not None and previous_rgb is not None:
                    try:
                        geometry = track_led_array(previous_rgb, rgb, geometry)
                    except RuntimeError as error:
                        print(f"LED tracking lost: {error}; automatically recalibrating")
                        geometry = None
                        baseline = None
                        unloaded_noise_sigma = None
                        baseline_collecting = False
                        no_contact_frame_count = 0
                        calibration_error = None
                        calibration_frames.clear()
                        baseline_samples.clear()
                        feature_history.clear()
                        lines.append("LED geometry: lost; automatically recalibrating")

                if geometry is None:
                    if calibration_error is None:
                        calibration_frames.append(rgb)
                        count = len(calibration_frames)
                        lines.append(
                            "LED geometry: calibrating "
                            f"{count}/{CALIBRATION_FRAME_COUNT} - hold camera still"
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
                                f"LED geometry: calibration failed: {calibration_error}",
                                "Reframe the five visible LEDs, then press r",
                            )
                        )
                    lines.append("Unloaded baseline: not set")
                else:
                    _draw_geometry(display, geometry)
                    lines.append("LED geometry: tracking")
                    features = brightest_red_features(rgb, geometry)
                    if baseline_collecting:
                        baseline_samples.append(features)
                        count = len(baseline_samples)
                        lines.append(
                            f"Unloaded baseline: collecting {count}/{BASELINE_FRAME_COUNT}"
                        )
                        if count == BASELINE_FRAME_COUNT:
                            baseline, unloaded_noise_sigma = (
                                unloaded_baseline_statistics(
                                    np.stack(baseline_samples),
                                )
                            )
                            baseline_collecting = False
                            no_contact_frame_count = 0
                            baseline_samples.clear()
                            feature_history.clear()
                            lines[-1] = "Unloaded baseline: ready"
                            print(
                                "unloaded baseline set: "
                                f"{np.round(baseline, 3).tolist()}"
                            )
                            print(
                                "unloaded feature noise sigma [DN]: "
                                f"{np.round(unloaded_noise_sigma, 3).tolist()}"
                            )
                    elif baseline is None:
                        lines.append("Unloaded baseline: not set - press b while unloaded")
                    else:
                        feature_history.append(features)
                        filtered_features = np.median(
                            np.stack(feature_history),
                            axis=0,
                        )
                        if unloaded_noise_sigma is None:
                            raise RuntimeError(
                                "unloaded baseline is missing its noise estimate"
                            )
                        estimate = estimate_contact_position(
                            filtered_features,
                            baseline,
                            unloaded_noise_sigma,
                            LED_POSITIONS_IN_IMAGE_ORDER_MM,
                        )
                        response = estimate.response
                        lines.append("Unloaded baseline: ready")
                        if not estimate.contact_detected:
                            lines.append("Contact: No contact")
                            no_contact_frame_count += 1
                            if (
                                no_contact_frame_count
                                >= NO_CONTACT_REANCHOR_FRAME_COUNT
                            ):
                                try:
                                    geometry = reanchor_led_array(rgb, geometry)
                                except RuntimeError as error:
                                    print(f"LED absolute re-anchor skipped: {error}")
                                else:
                                    feature_history.clear()
                                    lines.append("LED geometry: absolute re-anchor")
                                no_contact_frame_count = 0
                        else:
                            no_contact_frame_count = 0
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
                lines.append(
                    f"Frame {frame.frame_number} | {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
                    f"| {displayed_fps:.1f}/{CAMERA_FPS} FPS"
                )
                _draw_text(display, lines)
                visualization = _draw_response_panel(
                    display,
                    response,
                    unloaded_noise_sigma,
                )
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
                    unloaded_noise_sigma = None
                    baseline_collecting = False
                    no_contact_frame_count = 0
                    calibration_error = None
                    calibration_frames.clear()
                    baseline_samples.clear()
                    feature_history.clear()
                    print("recalibrating LED geometry")
                elif key == ord("b") and features is not None:
                    baseline = None
                    unloaded_noise_sigma = None
                    baseline_collecting = True
                    no_contact_frame_count = 0
                    baseline_samples.clear()
                    feature_history.clear()
                    print(
                        "collecting unloaded baseline: "
                        f"0/{BASELINE_FRAME_COUNT}"
                    )
                previous_rgb = rgb
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
