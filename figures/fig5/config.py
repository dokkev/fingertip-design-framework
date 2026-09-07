"""Fixed data and physical-coordinate contract for paper Figure 5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lumo.visualization import MATERIAL_LABELS, PAPER_LABELS


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


def _paper_name(material: str, morphology: str) -> str:
    return f"{MATERIAL_LABELS[material]} {PAPER_LABELS[morphology]}"


MORPHOLOGY_CONDITIONS = (
    MorphologyCondition(
        "solaris", "baseline", _paper_name("solaris", "baseline"), "Solaris-baseline"
    ),
    MorphologyCondition(
        "solaris",
        "flat_opt",
        _paper_name("solaris", "flat_opt"),
        "Solaris-flat-opt",
    ),
    MorphologyCondition(
        "solaris",
        "angled_opt",
        _paper_name("solaris", "angled_opt"),
        "Solaris-angled-opt",
    ),
    MorphologyCondition(
        "dragon_skin",
        "baseline",
        _paper_name("dragon_skin", "baseline"),
        "DragonSkin-baseline",
    ),
    MorphologyCondition(
        "dragon_skin",
        "flat_opt",
        _paper_name("dragon_skin", "flat_opt"),
        "DragonSkin-flat-opt",
    ),
    MorphologyCondition(
        "dragon_skin",
        "angled_opt",
        _paper_name("dragon_skin", "angled_opt"),
        "DragonSkin-angled-opt",
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

# These repeated acquisitions contain only the 30 mm indenter and retain their
# own unloaded references. They replace only the matching Figure 5 conditions.
ANALYSIS_CONDITION_OVERRIDES = {
    ("dragon_skin", "baseline", "sphere_30mm"): REPOSITORY_ROOT
    / "output"
    / "analysis"
    / "dragon_skin_baseline_30mm_rerun",
    ("dragon_skin", "angled_opt", "sphere_30mm"): REPOSITORY_ROOT
    / "output"
    / "analysis"
    / "dragon_skin_angled_opt_30mm_02",
}

COMPARISON_MORPHOLOGIES = ("baseline", "flat_opt", "angled_opt")
COMPARISON_CONDITIONS = (
    ("solaris", "sphere_10mm", "Solaris · 10 mm sphere"),
    ("solaris", "sphere_30mm", "Solaris · 30 mm sphere"),
    ("dragon_skin", "sphere_10mm", "Dragon Skin · 10 mm sphere"),
    ("dragon_skin", "sphere_30mm", "Dragon Skin · 30 mm sphere"),
)

# Panels (a), (b), and (c) use three shared morphology rows. Materials and
# indenter conditions run across columns, so every morphology can be followed
# horizontally without repeating six specimen rows.
MORPHOLOGY_TABLE_HEIGHT_RATIOS = (
    0.16,
    0.10,
    0.15,
    0.55,
    0.55,
    0.55,
)
MORPHOLOGY_TABLE_ROW_SLOTS = (3, 4, 5)
MORPHOLOGY_TABLE_HSPACE = 0.035
MATERIAL_SEPARATOR_COLOR = "#D4D4D4"
MATERIAL_SEPARATOR_LINEWIDTH_PT = 0.70

# The fixture has six measured stops at 10 mm spacing. Hole 1 is the distal
# stop and hole 6 is proximal. Coordinates are measured from the distal stop;
# this fixture spacing is independent of the fingertip's 11 mm LED pitch.
HOLE_TO_CONTACT_X_MM = {
    1: 0.0,
    2: 10.0,
    3: 20.0,
    4: 30.0,
    5: 40.0,
    6: 50.0,
}

# Three representative locations keep the raw-image atlas compact. The response
# fields and decoder continue to use all six acquired locations.
ATLAS_HOLES = (1, 3, 5)
ALL_HOLES = tuple(HOLE_TO_CONTACT_X_MM)

ATLAS_INDENTER = "sphere_10mm"
ATLAS_REPETITION = 1
ATLAS_TARGET_FORCE_N = 15.0
ATLAS_DISPLAY_EXPOSURE_EV_BY_MATERIAL = {
    "solaris": 0.2750,
    "dragon_skin": 0.5250,
}

# One camera-coordinate ROI is reused without recentering or photometric
# manipulation for every atlas frame. All Figure 5 sessions used the same
# 1920 x 1080 fixed-camera acquisition contract.
ATLAS_CROP_XYXY = (820, 170, 1170, 660)


def require_available_inputs() -> None:
    """Fail when a required Figure 5 session or analysis artifact is missing."""

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
    for condition, root in ANALYSIS_CONDITION_OVERRIDES.items():
        profile_path = root / "raw_data_summary" / "longitudinal_profiles.npz"
        metric_path = root / "results" / "morphology_metrics.csv"
        if not profile_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(
                f"missing required Figure 5 override for {condition}: {root}"
            )


__all__ = [
    "ALL_HOLES",
    "ANALYSIS_CONDITION_OVERRIDES",
    "ANALYSIS_ROOTS",
    "ATLAS_CROP_XYXY",
    "ATLAS_DISPLAY_EXPOSURE_EV_BY_MATERIAL",
    "ATLAS_HOLES",
    "ATLAS_INDENTER",
    "ATLAS_REPETITION",
    "ATLAS_TARGET_FORCE_N",
    "COMPARISON_CONDITIONS",
    "COMPARISON_MORPHOLOGIES",
    "FIGURE_DIRECTORY",
    "HOLE_TO_CONTACT_X_MM",
    "MATERIAL_SEPARATOR_COLOR",
    "MATERIAL_SEPARATOR_LINEWIDTH_PT",
    "MORPHOLOGY_CONDITIONS",
    "MORPHOLOGY_TABLE_HEIGHT_RATIOS",
    "MORPHOLOGY_TABLE_HSPACE",
    "MORPHOLOGY_TABLE_ROW_SLOTS",
    "MorphologyCondition",
    "REPOSITORY_ROOT",
    "require_available_inputs",
]
