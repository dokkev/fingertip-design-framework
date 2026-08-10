"""Trace and plot deterministic no-load light transport in the fingertip."""

from __future__ import annotations

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

repository_root = ensure_repository_root()

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.fingertip_sensor_model import FingertipSensorModel
from optics.cross_section import trace_no_load_sensor
from visualization.optical_cross_section import save_cross_section_transport_figure


def main() -> int:
    geometry = FingertipModel(
        FingertipParameters(void_width=1.0, void_height=2.0)
    )
    sensor = FingertipSensorModel.from_geometry(geometry)
    domain, result = trace_no_load_sensor(sensor)
    output = save_cross_section_transport_figure(
        sensor,
        domain,
        result,
        repository_root / "output" / "optics" / "no_load_light_transport_2d.png",
        title="No-load 2D light transport",
    )
    print(output)
    print(
        "rays={rays} segments={segments} launched={launched:.6f} "
        "escaped={escaped:.6f} absorbed={absorbed:.6f} "
        "terminated={terminated:.6f}".format(
            rays=result.launched_ray_count,
            segments=len(result.segments),
            launched=result.launched_weight,
            escaped=result.escaped_weight,
            absorbed=result.absorbed_weight,
            terminated=result.terminated_weight,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
