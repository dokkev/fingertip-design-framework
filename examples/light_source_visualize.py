"""Show the fingertip LED/source schematic and render a Mitsuba baseline."""

from __future__ import annotations

import matplotlib.pyplot as plt

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

repository_root = ensure_repository_root()

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.fingertip_sensor_model import FingertipSensorModel
from optics.mitsuba_reference import render_point_source_spread
from visualization.geometry import plot_fingertip


def main() -> int:
    model = FingertipModel(FingertipParameters())
    sensor = FingertipSensorModel.from_geometry(model)
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    plot_fingertip(
        model,
        sensor_model=sensor,
        ax=axis,
        show_void=True,
        show_light_source=True,
    )
    output = render_point_source_spread(
        sensor,
        repository_root / "output" / "optics" / "point_source_spread.png",
    )
    print(output)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
