"""Visualize image-derived fingertip boundary candidates from a D435."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.hardware import RealSenseColorCamera  # noqa: E402
from experiments.localization import (  # noqa: E402
    detect_fingertip_boundary,
    detect_led_array,
)


CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CAMERA_WARMUP_FRAME_COUNT = 30
CAMERA_FRAME_TIMEOUT_MS = 2000
WINDOW_NAME = "LUMO fingertip boundary"


def _panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.putText(
        panel,
        title,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        title,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    camera = RealSenseColorCamera(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        with camera:
            for _ in range(CAMERA_WARMUP_FRAME_COUNT):
                camera.read(timeout_ms=CAMERA_FRAME_TIMEOUT_MS)
            print(
                f"camera: {camera.device_name}, serial={camera.serial_number}, "
                f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}"
            )
            print("boundary-only viewer; q/esc=quit")
            while True:
                frame = camera.read(timeout_ms=CAMERA_FRAME_TIMEOUT_MS)
                rgb = frame.rgb
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                try:
                    boundary = detect_fingertip_boundary(rgb)
                except RuntimeError as error:
                    cyan_view = np.zeros_like(bgr)
                    boundary_view = bgr.copy()
                    cv2.putText(
                        boundary_view,
                        str(error),
                        (30, max(150, round(0.14 * boundary_view.shape[0]))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cyan_view = np.zeros_like(bgr)
                    cyan_view[boundary.search_mask] = (180, 180, 180)
                    boundary_view = bgr.copy()
                    band_overlay = boundary_view.copy()
                    band_overlay[boundary.search_mask] = (255, 170, 0)
                    boundary_view = cv2.addWeighted(
                        boundary_view,
                        0.78,
                        band_overlay,
                        0.22,
                        0.0,
                    )
                    dorsal = np.rint(boundary.dorsal_boundary_xy_px).astype(np.int32)
                    palmar = np.rint(boundary.palmar_boundary_xy_px).astype(np.int32)
                    cv2.polylines(
                        boundary_view,
                        [dorsal],
                        False,
                        (255, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.polylines(
                        boundary_view,
                        [palmar],
                        False,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    dorsal_support_px = (
                        boundary.core_y_span[1] - boundary.core_y_span[0]
                    )
                    cv2.putText(
                        boundary_view,
                        (
                            f"pad width={boundary.estimated_pad_width_px:.1f} px  "
                            f"dorsal support={dorsal_support_px} px"
                        ),
                        (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    try:
                        geometry = detect_led_array(
                            rgb,
                            search_mask=boundary.search_mask,
                        )
                    except RuntimeError as error:
                        cv2.putText(
                            boundary_view,
                            f"LED detection: {error}",
                            (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        for polygon in geometry.roi_polygons_xy_px:
                            cv2.polylines(
                                boundary_view,
                                [np.rint(polygon).astype(np.int32)],
                                True,
                                (0, 170, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        for landmark in geometry.landmarks_xy_px:
                            cv2.circle(
                                boundary_view,
                                tuple(np.rint(landmark).astype(int)),
                                4,
                                (0, 255, 0),
                                -1,
                                cv2.LINE_AA,
                            )

                display = np.hstack(
                    (
                        _panel(bgr, "RGB"),
                        _panel(cyan_view, "fingertip search mask"),
                        _panel(boundary_view, "dorsal / palmar / LED ROIs"),
                    )
                )
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
