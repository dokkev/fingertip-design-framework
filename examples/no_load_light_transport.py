"""Render the no-load zero-displacement state with persistent Mitsuba APIs."""

from __future__ import annotations

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

repository_root = ensure_repository_root()

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.fingertip_sensor_model import FingertipSensorModel
from optics.adapters import build_preview_pad_mesh_template
from optics.geometry import ExtrudedOpticalMeshTemplate, PadDeformationState2D
from optics.mitsuba import (
    MitsubaRenderSession,
    MitsubaRenderSettings,
    default_cross_section_camera,
)
from visualization.camera_response import save_camera_render


def main() -> int:
    geometry = FingertipModel(FingertipParameters())
    sensor = FingertipSensorModel.from_geometry(geometry)
    template = build_preview_pad_mesh_template(sensor)
    zero_state = PadDeformationState2D.zero(template)
    settings = MitsubaRenderSettings()
    extrusion = ExtrudedOpticalMeshTemplate.from_pad_mesh(
        template,
        depth_mm=settings.optical_depth_mm,
    )
    session = MitsubaRenderSession(
        sensor_model=sensor,
        mesh_template=template,
        extrusion=extrusion,
        camera=default_cross_section_camera(sensor, template),
        settings=settings,
    )
    result = session.render_state(zero_state)
    output = save_camera_render(
        result,
        repository_root / "output" / "optics" / "no_load_light_transport.png",
        gamma=2.2,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
