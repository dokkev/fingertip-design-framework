"""Run the discrete 0.5 mm LUMO multi-objective BO campaign."""

from pathlib import Path

from lumo.optimization.ax_bo import run


# User settings. Bounds are physical millimeters on the 0.5 mm lattice.
PARAMETER_BOUNDS_MM = {
    "flat_pad_height_mm": (2.0, 29.0),
    "semiellipse_height_mm": (1.0, 20.0),
    "stem_width_mm": (4.0, 15.0),
    "stem_height_mm": (2.0, 15.0),
    "void_width_mm": (0.0, 4.0),
    "void_height_mm": (0.0, 5.0),
}
# Packaged URDFs share one initial pose based on a 20 mm reference indenter.
INDENTER_URDFS = (
    "sphere_10mm.urdf",
    "sphere_15mm.urdf",
    "sphere_20mm.urdf",
)
FORCE_TARGETS_N = (10.0, 15.0, 20.0)
SETTLE_DURATION_S = 3.0
FORCE_TOLERANCE_FRACTION = 0.20  # ±this fraction of each force target
TARGET_MORPHOLOGIES = 120
OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "optimization"
    / "mobo_discrete_05mm_clean"
)


def main() -> None:
    run(
        output_directory=OUTPUT_DIRECTORY,
        target_bo_trials=TARGET_MORPHOLOGIES,
        campaign_name="discrete-05mm",
        parameter_bounds_mm=PARAMETER_BOUNDS_MM,
        indenter_urdfs=INDENTER_URDFS,
        force_targets_n=FORCE_TARGETS_N,
        settle_duration_s=SETTLE_DURATION_S,
        force_tolerance_fraction=FORCE_TOLERANCE_FRACTION,
    )


if __name__ == "__main__":
    main()
