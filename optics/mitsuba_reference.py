"""Minimal free-space Mitsuba point-source reference render."""

from __future__ import annotations

from pathlib import Path

from model.fingertip_sensor_model import FingertipSensorModel


def render_point_source_spread(
    sensor_model: FingertipSensorModel,
    output_path: str | Path,
    *,
    spp: int = 128,
) -> Path:
    """Render a free-space point source onto a diffuse y=0 receiver plane."""
    try:
        import mitsuba as mi
    except ImportError as exc:
        raise RuntimeError(
            "Mitsuba is required for the optical reference render; install "
            "the optional 'optics' dependency."
        ) from exc

    mi.set_variant("scalar_rgb")
    from mitsuba import ScalarTransform4f

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = sensor_model.led_source_position_3d
    source_power = sensor_model.led.relative_radiant_power
    receiver_to_world = ScalarTransform4f.rotate([1.0, 0.0, 0.0], 90.0) @ (
        ScalarTransform4f.scale([8.0, 8.0, 8.0])
    )
    camera_to_world = ScalarTransform4f.look_at(
        origin=[0.0, -12.0, 0.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
    )
    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 8},
            "light": {
                "type": "point",
                "position": list(source),
                "intensity": {
                    "type": "rgb",
                    "value": [
                        40.0 * source_power,
                        16.0 * source_power,
                        4.0 * source_power,
                    ],
                },
            },
            "receiver": {
                "type": "rectangle",
                "to_world": receiver_to_world,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": [0.65, 0.68, 0.72],
                    },
                },
            },
            "sensor": {
                "type": "perspective",
                "fov": 42.0,
                "to_world": camera_to_world,
                "film": {
                    "type": "hdrfilm",
                    "width": 512,
                    "height": 512,
                    "rfilter": {"type": "box"},
                },
            },
        }
    )
    image = mi.render(scene, spp=spp)
    mi.util.write_bitmap(str(output), image)
    return output
