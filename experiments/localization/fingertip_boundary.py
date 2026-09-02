"""Public facade for image-derived fingertip boundaries."""

from __future__ import annotations

import numpy as np

from .fingertip_segmentation import (
    FingertipBoundaryRegion,
    segment_fingertip,
)


def detect_fingertip_boundary(rgb: np.ndarray) -> FingertipBoundaryRegion:
    """Detect the smooth emissive silicone silhouette from one RGB frame."""

    return segment_fingertip(rgb).region


__all__ = ["FingertipBoundaryRegion", "detect_fingertip_boundary"]
