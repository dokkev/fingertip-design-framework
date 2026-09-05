"""Fixed data and physical-coordinate contract for paper Figure 5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIRECTORY = Path(__file__).resolve().parent
DATASET_ROOT = REPOSITORY_ROOT / "output" / "contact_dataset"


@dataclass(frozen=True)
class MorphologyCondition:
    """One fabricated specimen row in Figure 5."""

    material: str
    morphology: str
    display_name: str
    session_directory: str | None

    @property
    def session_path(self) -> Path | None:
        if self.session_directory is None:
            return None
        return DATASET_ROOT / self.session_directory

    @property
    def pending(self) -> bool:
        return self.session_directory is None


MORPHOLOGY_CONDITIONS = (
    MorphologyCondition(
        "solaris", "baseline", "Solaris Baseline", "2026-09-04_solaris_baseline_01"
    ),
    MorphologyCondition(
        "solaris", "flat_opt", "Solaris Flat-opt", "2026-09-04_solaris_flat_opt_01"
    ),
    MorphologyCondition(
        "solaris",
        "angled_opt",
        "Solaris Angled-opt",
        "2026-09-04_solaris_angled_opt_01",
    ),
    MorphologyCondition(
        "dragon_skin",
        "baseline",
        "Dragon Skin Baseline",
        "2026-09-04_dragon_skin_baseline",
    ),
    MorphologyCondition(
        "dragon_skin",
        "flat_opt",
        "Dragon Skin Flat-opt",
        "2026-09-04_dragon_skin_flat_opt",
    ),
    MorphologyCondition(
        "dragon_skin", "angled_opt", "Dragon Skin Angled-opt", None
    ),
)

ANALYSIS_ROOTS = {
    "solaris": REPOSITORY_ROOT
    / "output"
    / "analysis"
    / "solaris_01_morphology_comparison",
    "dragon_skin": REPOSITORY_ROOT
    / "output"
    / "analysis"
    / "dragon_skin_morphology_comparison",
}

COMPARISON_MORPHOLOGIES = ("baseline", "flat_opt", "angled_opt")
COMPARISON_TITLES = ("Baseline", "Flat-opt", "Angled-opt")
COMPARISON_CONDITIONS = (
    ("solaris", "sphere_10mm", "Solaris · 10 mm sphere"),
    ("solaris", "sphere_30mm", "Solaris · 30 mm sphere"),
    ("dragon_skin", "sphere_10mm", "Dragon Skin · 10 mm sphere"),
    ("dragon_skin", "sphere_30mm", "Dragon Skin · 30 mm sphere"),
)

# Panels (a) and (b) use this exact vertical grammar so matching morphology
# rows align across the final composed figure.  The narrow empty row separates
# the two materials without adding a decorative rule.
MORPHOLOGY_TABLE_HEIGHT_RATIOS = (0.24, 0.20, 1.0, 1.0, 1.0, 0.10, 1.0, 1.0, 1.0)
MORPHOLOGY_TABLE_ROW_SLOTS = (2, 3, 4, 6, 7, 8)
MORPHOLOGY_TABLE_HSPACE = 0.018

# The fixture has six equally spaced stops across its 55 mm travel. Hole 1 is
# the distal stop and hole 6 is proximal. Coordinates are measured from the
# distal stop, so the acquisition labels map to physical positions explicitly.
HOLE_TO_CONTACT_X_MM = {
    1: 0.0,
    2: 11.0,
    3: 22.0,
    4: 33.0,
    5: 44.0,
    6: 55.0,
}

# Five consecutive locations keep the raw-image atlas legible. The response
# fields and transfer metric continue to use all six acquired locations.
ATLAS_HOLES = (1, 2, 3, 4, 5)
ALL_HOLES = tuple(HOLE_TO_CONTACT_X_MM)

ATLAS_INDENTER = "sphere_10mm"
ATLAS_REPETITION = 1
ATLAS_TARGET_FORCE_N = 15.0
ATLAS_DISPLAY_EXPOSURE_EV = 0.0

# One camera-coordinate ROI is reused without recentering or photometric
# manipulation for every atlas frame. All Figure 5 sessions used the same
# 1920 x 1080 fixed-camera acquisition contract.
ATLAS_CROP_XYXY = (820, 170, 1170, 750)


def require_available_inputs() -> None:
    """Fail on missing required data while allowing the one pending specimen."""

    for condition in MORPHOLOGY_CONDITIONS:
        if condition.pending:
            continue
        assert condition.session_path is not None
        if not (condition.session_path / "session.json").is_file():
            raise FileNotFoundError(
                f"missing required Figure 5 session: {condition.session_path}"
            )
    for material, root in ANALYSIS_ROOTS.items():
        path = root / "raw_data_summary" / "longitudinal_profiles.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing required {material} analysis artifact: {path}"
            )


__all__ = [
    "ALL_HOLES",
    "ANALYSIS_ROOTS",
    "ATLAS_CROP_XYXY",
    "ATLAS_DISPLAY_EXPOSURE_EV",
    "ATLAS_HOLES",
    "ATLAS_INDENTER",
    "ATLAS_REPETITION",
    "ATLAS_TARGET_FORCE_N",
    "COMPARISON_CONDITIONS",
    "COMPARISON_MORPHOLOGIES",
    "COMPARISON_TITLES",
    "FIGURE_DIRECTORY",
    "HOLE_TO_CONTACT_X_MM",
    "MORPHOLOGY_CONDITIONS",
    "MORPHOLOGY_TABLE_HEIGHT_RATIOS",
    "MORPHOLOGY_TABLE_HSPACE",
    "MORPHOLOGY_TABLE_ROW_SLOTS",
    "MorphologyCondition",
    "REPOSITORY_ROOT",
    "require_available_inputs",
]
