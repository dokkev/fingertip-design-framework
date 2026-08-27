"""Run the discrete 0.5 mm LUMO multi-objective BO campaign."""

import logging
import os
import warnings
from pathlib import Path

from lumo.optimization.ax_bo import run


# User settings. Available mechanics: silicone.
# Available optics: solaris_{low,nominal,high} and
# dragon_skin_10_nv_{low,nominal,high}.
MECHANICS_PRESET = "silicone"
OPTICAL_PRESET = "dragon_skin_10_nv_nominal"
OTK_INCLUDE_DIR = (
    Path(__file__).resolve().parents[2] / "optix-toolkit" / "ShaderUtil" / "include"
)
# Bounds are physical millimeters on the 0.5 mm lattice.
PARAMETER_BOUNDS_MM = {
    "flat_pad_height_mm": (2.0, 29.0),
    "semiellipse_height_mm": (1.0, 20.0),
    "stem_width_mm": (4.0, 15.0),
    "stem_height_mm": (2.0, 15.0),
    "void_width_mm": (0.0, 10.0),
}
INDENTER_URDFS = (
    "sphere_10mm.urdf",
    "sphere_15mm.urdf",
    "sphere_20mm.urdf",
)
SPHERE_DIAMETERS_MM = (10.0, 15.0, 20.0)
CONTACT_Y_MM = (-22.0, -11.0, -5.5, 0.0, 5.5, 11.0, 22.0)
INITIAL_CLEARANCE_M = 1.0e-3
FORCE_TARGETS_N = (1.0, 2.0, 5.0, 10.0)
TARGET_MORPHOLOGIES = 120
OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "optimization"
    / "mobo_fingertip_instantaneous_05mm"
)


def _quiet_third_party_output() -> None:
    """Keep campaign progress while suppressing known dependency noise."""
    import warp as wp
    from ax.exceptions.core import AxOptimizationWarning
    from ax.utils.common.logger import set_ax_logger_levels

    set_ax_logger_levels(logging.WARNING)
    wp.config.log_level = wp.LOG_WARNING
    warnings.filterwarnings(
        "ignore",
        message=(
            r"Encountered a `MultiObjective` without objective thresholds\."
        ),
        category=AxOptimizationWarning,
        module=r"ax\.adapter\.transforms\.winsorize",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"To copy construct from a tensor, it is recommended to use",
        category=UserWarning,
        module=r"ax\.generators\.torch\.botorch_moo_utils",
    )


def main() -> None:
    _quiet_third_party_output()
    os.environ.setdefault("OTK_INCLUDE_DIR", str(OTK_INCLUDE_DIR))
    run(
        output_directory=OUTPUT_DIRECTORY,
        target_bo_trials=TARGET_MORPHOLOGIES,
        mechanics_preset=MECHANICS_PRESET,
        optical_preset=OPTICAL_PRESET,
        parameter_bounds_mm=PARAMETER_BOUNDS_MM,
        indenter_urdfs=INDENTER_URDFS,
        sphere_diameters_mm=SPHERE_DIAMETERS_MM,
        contact_y_mm=CONTACT_Y_MM,
        initial_clearance_m=INITIAL_CLEARANCE_M,
        force_targets_n=FORCE_TARGETS_N,
    )


if __name__ == "__main__":
    main()
