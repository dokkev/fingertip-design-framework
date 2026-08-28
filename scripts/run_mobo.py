"""Run the discrete 0.5 mm LUMO multi-objective BO campaign."""

import logging
import os
import warnings
from pathlib import Path

from lumo.optimization.ax_bo import run


# User settings. Available mechanics: silicone and solaris.
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
    "void_width_mm": (0.0, 7.5),
}
# In order: flat-pad height, semiellipse height, stem width, stem height,
# and void width [mm]. These old Dragon-campaign designs are re-evaluated
# normally under this campaign's 75-scenario objective; no old objective value
# is imported.
INITIAL_MORPHOLOGIES_MM = (
    (14.5, 4.0, 5.0, 12.5, 5.0),  # old trial 117
    (13.5, 1.5, 7.0, 9.5, 5.5),  # old trial 19
    (5.5, 2.0, 9.0, 2.0, 0.5),  # old trial 128
    (13.0, 7.0, 7.5, 14.5, 0.0),  # old trial 49
    (2.5, 17.5, 7.0, 2.0, 3.5),  # old trial 6
)
INDENTER_URDFS = (
    "sphere_10mm.urdf",
    "sphere_15mm.urdf",
    "sphere_20mm.urdf",
)
SPHERE_DIAMETERS_MM = (10.0, 15.0, 20.0)
# Physical fingertip rotations about world +Y. Scenario count is
# len(spheres) * len(angles) * len(contact Y locations).
INDENTATION_ANGLES_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
CONTACT_Y_MM = (-11.0, -5.5, 0.0, 5.5, 11.0)
# The angled trajectories share a conservative pre-contact starting distance.
INITIAL_CLEARANCE_M = 10.0e-3
FORCE_TARGETS_N = (1.0, 2.0, 5.0, 10.0)
TARGET_MORPHOLOGIES = 120
OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "optimization"
    / "mobo_fingertip_orientation_robust_1_2_5_10_05mm"
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
        indentation_angles_deg=INDENTATION_ANGLES_DEG,
        contact_y_mm=CONTACT_Y_MM,
        initial_clearance_m=INITIAL_CLEARANCE_M,
        force_targets_n=FORCE_TARGETS_N,
        initial_morphologies_mm=INITIAL_MORPHOLOGIES_MM,
    )


if __name__ == "__main__":
    main()
