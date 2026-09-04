"""Image-space contour-displacement measurements for a fixed experiment pose."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ContourReference:
    """Fixed unloaded contour samples and their local image normals."""

    points_xy: np.ndarray
    normals_xy: np.ndarray
    reference_offsets_px: np.ndarray
    image_shape: tuple[int, int]
    search_radius_px: int


def build_contour_reference(
    reference_rgb: np.ndarray,
    reference_mask: np.ndarray,
    *,
    sample_count: int = 384,
    search_radius_px: int = 12,
) -> ContourReference:
    """Prepare fixed contour samples; no loaded frame changes this geometry."""

    image = np.asarray(reference_rgb)
    mask = np.asarray(reference_mask, dtype=bool)
    if image.shape != (*mask.shape, 3) or image.dtype != np.uint8:
        raise ValueError("reference image and mask must have matching uint8 RGB shape")
    if sample_count < 16 or search_radius_px < 1:
        raise ValueError("sample_count must be >= 16 and search_radius_px positive")
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    indices = np.linspace(
        0, len(contour), min(sample_count, len(contour)), endpoint=False
    )
    points = contour[np.floor(indices).astype(int)]
    previous = np.roll(points, 2, axis=0)
    following = np.roll(points, -2, axis=0)
    tangents = following - previous
    lengths = np.linalg.norm(tangents, axis=1)
    valid = lengths > np.finfo(np.float64).eps
    points = points[valid]
    tangents = tangents[valid] / lengths[valid, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    offsets, _ = _edge_offsets(reference_rgb, points, normals, search_radius_px)
    return ContourReference(points, normals, offsets, mask.shape, search_radius_px)


def contour_deformation(
    loaded_rgb: np.ndarray,
    reference: ContourReference,
) -> dict[str, float | str | bool]:
    """Measure loaded edge motion along fixed unloaded contour normals."""

    image = np.asarray(loaded_rgb)
    if image.shape != (*reference.image_shape, 3) or image.dtype != np.uint8:
        return _invalid("loaded image shape does not match contour calibration")
    offsets, strengths = _edge_offsets(
        image,
        reference.points_xy,
        reference.normals_xy,
        reference.search_radius_px,
    )
    valid = np.isfinite(offsets) & np.isfinite(reference.reference_offsets_px)
    if np.count_nonzero(valid) < 16 or not np.any(strengths[valid] > 0.0):
        return _invalid("too few visible contour-edge samples")
    displacement = np.abs(offsets[valid] - reference.reference_offsets_px[valid])
    return {
        "deformation_valid": True,
        "deformation_invalid_reason": "",
        "deformation_rms_px": float(np.sqrt(np.mean(displacement**2))),
        "deformation_p95_px": float(np.percentile(displacement, 95.0)),
        "deformation_max_px": float(np.max(displacement)),
        "deformation_sample_count": int(len(displacement)),
    }


def _edge_offsets(
    rgb: np.ndarray,
    points_xy: np.ndarray,
    normals_xy: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    candidates = np.arange(-radius, radius + 1, dtype=np.float32)
    sample_x = points_xy[:, 0, None] + normals_xy[:, 0, None] * candidates
    sample_y = points_xy[:, 1, None] + normals_xy[:, 1, None] * candidates
    gx = cv2.remap(
        gradient_x,
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    gy = cv2.remap(
        gradient_y,
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    normal_response = np.abs(gx * normals_xy[:, 0, None] + gy * normals_xy[:, 1, None])
    maxima = np.argmax(normal_response, axis=1)
    return candidates[maxima].astype(np.float64), normal_response[
        np.arange(len(maxima)), maxima
    ].astype(np.float64)


def _invalid(reason: str) -> dict[str, float | str | bool]:
    return {
        "deformation_valid": False,
        "deformation_invalid_reason": reason,
        "deformation_rms_px": float("nan"),
        "deformation_p95_px": float("nan"),
        "deformation_max_px": float("nan"),
        "deformation_sample_count": 0,
    }


__all__ = ["ContourReference", "build_contour_reference", "contour_deformation"]
