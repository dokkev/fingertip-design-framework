"""Dependency-light architecture check for a synthetic loaded optical state."""

from __future__ import annotations

import numpy as np

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

repository_root = ensure_repository_root()

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.fingertip_sensor_model import FingertipSensorModel
from optics.adapters import build_preview_pad_mesh_template
from optics.cross_section import CrossSectionTraceSettings, trace_pad_state
from optics.geometry import PadDeformationState2D
from visualization.optical_cross_section import save_cross_section_transport_figure


def main() -> int:
    geometry = FingertipModel(FingertipParameters())
    sensor = FingertipSensorModel.from_geometry(geometry)
    template = build_preview_pad_mesh_template(sensor)
    displacement_magnitude_mm = 0.25
    displacement = np.zeros_like(template.reference_coordinates_mm)
    bottom_nodes = template.boundary_node_indices_for("pad_cutout_bottom")
    displacement[bottom_nodes, 1] = -displacement_magnitude_mm
    loaded_state = PadDeformationState2D(
        displacement_mm=displacement,
        metadata={
            "condition": "synthetic_loaded",
            "description": "pad cutout bottom displaced distally",
        },
    )
    template.validate_state(loaded_state)

    settings = CrossSectionTraceSettings(
        ray_count=81,
        grid_width=120,
        grid_height=120,
        maximum_segment_count=10000,
    )
    domain, result = trace_pad_state(
        sensor,
        template,
        loaded_state,
        settings,
    )
    output = save_cross_section_transport_figure(
        sensor,
        domain,
        result,
        repository_root
        / "output"
        / "optics"
        / "synthetic_loaded_light_transport_2d.png",
        title="Synthetic loaded 2D light transport",
    )
    print(f"source_position_mm={sensor.led_source_position_2d}")
    print(
        "cutout_bottom_displacement_mm="
        f"{(0.0, -displacement_magnitude_mm)}"
    )
    print(f"tagged_boundary_group_count={len(template.semantic_boundary_tags)}")
    print(f"primary_ray_count={result.launched_ray_count}")
    print(f"segment_count={len(result.segments)}")
    print(
        "launched={launched:.6f} escaped={escaped:.6f} "
        "absorbed={absorbed:.6f} terminated={terminated:.6f}".format(
            launched=result.launched_weight,
            escaped=result.escaped_weight,
            absorbed=result.absorbed_weight,
            terminated=result.terminated_weight,
        )
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
