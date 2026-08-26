"""Run the discrete 0.5 mm LUMO multi-objective BO campaign."""

import os
from pathlib import Path

from lumo.optimization.ax_bo import run


# User settings. Available mechanics: silicone.
# Available optics: solaris_{low,nominal,high} and
# dragon_skin_10_nv_{low,nominal,high}.
VISCOELASTIC_PRESET = "silicone"
OPTICAL_PRESET = "solaris_nominal"
OTK_INCLUDE_DIR = (
    Path(__file__).resolve().parents[2] / "optix-toolkit" / "ShaderUtil" / "include"
)
# Bounds are physical millimeters on the 0.5 mm lattice.
PARAMETER_BOUNDS_MM = {
    "flat_pad_height_mm": (2.0, 29.0),
    "semiellipse_height_mm": (1.0, 20.0),
    "stem_width_mm": (4.0, 15.0),
    "stem_height_mm": (2.0, 15.0),
    "void_width_mm": (0.0, 4.0),
}
INDENTER_URDFS = (
    "sphere_5mm.urdf",
    "sphere_10mm.urdf",
    "sphere_20mm.urdf",
)
SPHERE_DIAMETERS_MM = (5.0, 10.0, 20.0)
CONTACT_Y_MM = (-22.0, -11.0, -5.5, 0.0, 5.5, 11.0, 22.0)
INITIAL_CLEARANCE_M = 1.0e-3
FORCE_TARGETS_N = (5.0, 10.0, 15.0, 20.0)
SETTLE_DURATION_S = 5.0
FORCE_TOLERANCE_FRACTION = 0.10  # ±this fraction of each force target
TARGET_MORPHOLOGIES = 120
OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "optimization"
    / "mobo_full_finger_05mm"
)


def main() -> None:
    os.environ.setdefault("OTK_INCLUDE_DIR", str(OTK_INCLUDE_DIR))
    run(
        output_directory=OUTPUT_DIRECTORY,
        target_bo_trials=TARGET_MORPHOLOGIES,
        campaign_name="discrete-05mm",
        viscoelastic_preset=VISCOELASTIC_PRESET,
        optical_preset=OPTICAL_PRESET,
        parameter_bounds_mm=PARAMETER_BOUNDS_MM,
        indenter_urdfs=INDENTER_URDFS,
        sphere_diameters_mm=SPHERE_DIAMETERS_MM,
        contact_y_mm=CONTACT_Y_MM,
        initial_clearance_m=INITIAL_CLEARANCE_M,
        force_targets_n=FORCE_TARGETS_N,
        settle_duration_s=SETTLE_DURATION_S,
        force_tolerance_fraction=FORCE_TOLERANCE_FRACTION,
    )


if __name__ == "__main__":
    main()
