from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

from model import Fingertip, FingertipParameters
from optics import TraceSettings, trace


@pytest.mark.gmsh
def test_loaded_cutout_gap_starts_in_air_and_reaches_silicone() -> None:
    tip = Fingertip(FingertipParameters())
    mesh = tip.mesh()
    settings = TraceSettings(
        ray_count=31,
        grid_width=48,
        grid_height=48,
        maximum_segment_count=5000,
    )
    displacement = np.zeros_like(mesh.coordinates)
    cutout_bottom = mesh.boundary_node_indices_for("pad_cutout_bottom")
    displacement[cutout_bottom, 1] = -0.05

    loaded = trace(tip, mesh.deformed(displacement), settings)

    assert loaded.segments[0].medium == "air"
    assert any(segment.medium == "silicone" for segment in loaded.segments)
    assert loaded.air_region.covers(
        Point(tip.led_source[0], tip.led_source[1] - 0.025)
    )
    tolerance = tip.parameters.geometry_tolerance
    assert loaded.outer_envelope.buffer(tolerance).covers(
        loaded.silicone_region
    )
