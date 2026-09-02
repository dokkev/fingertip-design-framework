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
from experiments.localization import detect_led_array  # noqa: E402
from experiments.localization.fingertip_boundary import (  # noqa: E402
    _detect_fingertip_boundary_with_diagnostics,
)


CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CAMERA_WARMUP_FRAME_COUNT = 30
CAMERA_FRAME_TIMEOUT_MS = 2000
WINDOW_NAME = "LUMO fingertip boundary"


def _panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = cv2.resize(image, (384, 216), interpolation=cv2.INTER_AREA)
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


def _mask_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = bgr.copy()
    overlay[mask] = color
    return cv2.addWeighted(bgr, 0.72, overlay, 0.28, 0.0)


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
                    diagnostics = _detect_fingertip_boundary_with_diagnostics(rgb)
                except RuntimeError as error:
                    coarse_view = np.zeros_like(bgr)
                    raw_view = np.zeros_like(bgr)
                    final_view = np.zeros_like(bgr)
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
                    boundary = diagnostics.region
                    coarse_view = _mask_overlay(
                        bgr,
                        diagnostics.coarse_prior_mask,
                        (255, 150, 40),
                    )
                    raw_view = _mask_overlay(
                        bgr,
                        diagnostics.raw_component_mask,
                        (40, 170, 255),
                    )
                    final_view = _mask_overlay(
                        bgr,
                        diagnostics.final_mask,
                        (70, 190, 70),
                    )
                    boundary_view = bgr.copy()
                    cv2.polylines(
                        boundary_view,
                        [np.rint(diagnostics.contour_xy_px).astype(np.int32)],
                        True,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        boundary_view,
                        (
                            f"pad width={boundary.estimated_pad_width_px:.1f} px  "
                            f"area={np.count_nonzero(diagnostics.final_mask)} px  "
                            f"scale={diagnostics.geometry_scale:.3f}  "
                            f"runtime={diagnostics.runtime_ms:.0f} ms"
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
                        _panel(coarse_view, "coarse paired-LSD prior"),
                        _panel(raw_view, "raw GrabCut component"),
                        _panel(final_view, "emissive fingertip mask"),
                        _panel(boundary_view, "smooth contour / LED ROIs"),
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
