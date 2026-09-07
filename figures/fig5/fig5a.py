"""Figure 5(a): raw fixed-camera optical-signature atlas."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import SubplotSpec  # noqa: E402

from experiments.analysis.dataset import index_session  # noqa: E402
from experiments.analysis.metrics import actual_force_magnitude  # noqa: E402
from experiments.analysis.optical import load_rgb  # noqa: E402
from lumo.visualization import DEFAULT_STYLE, publication_context, save_figure  # noqa: E402

from .config import (  # noqa: E402
    ATLAS_CROP_XYXY,
    ATLAS_DISPLAY_EXPOSURE_EV_BY_MATERIAL,
    ATLAS_HOLES,
    ATLAS_INDENTER,
    ATLAS_REPETITION,
    ATLAS_TARGET_FORCE_N,
    FIGURE_DIRECTORY,
    HOLE_TO_CONTACT_X_MM,
    MORPHOLOGY_CONDITIONS,
    MORPHOLOGY_TABLE_HEIGHT_RATIOS,
    MORPHOLOGY_TABLE_HSPACE,
    MORPHOLOGY_TABLE_ROW_SLOTS,
    REPOSITORY_ROOT,
    require_available_inputs,
)


@dataclass(frozen=True)
class AtlasSelection:
    """One auditable raw frame selected for the atlas."""

    material: str
    morphology: str
    display_name: str
    displayed_contact_position_mm: float | None
    hole_index: int | None
    repetition_index: int | None
    target_force_n: float
    actual_force_n: float
    image_path: Path
    timestamp_s: float
    unloaded: bool
    run_id: str
    capture_id: str


def _force(frame: object) -> float:
    measurements = frame.measurements
    return actual_force_magnitude(
        float(measurements["Fx_N"]),
        float(measurements["Fy_N"]),
        float(measurements["Fz_N"]),
    )


def _time(frame: object) -> float:
    return float(frame.measurements["camera_host_time_s"])


def _select_loaded_frames(condition: object) -> list[AtlasSelection]:
    assert condition.session_path is not None
    index = index_session(condition.session_path, expected_repetitions=5)
    selected: list[AtlasSelection] = []
    for hole in ATLAS_HOLES:
        candidates = [
            frame
            for frame in index.frames
            if frame.run is not None
            and frame.run.status == "complete"
            and frame.run.indenter == ATLAS_INDENTER
            and frame.run.hole_index == hole
            and frame.run.repetition_index == ATLAS_REPETITION
            and frame.target_force_n is not None
            and np.isclose(frame.target_force_n, ATLAS_TARGET_FORCE_N)
        ]
        if not candidates:
            raise RuntimeError(
                f"missing required loaded atlas frame for {condition.display_name}, "
                f"{ATLAS_INDENTER}, hole {hole}, repetition {ATLAS_REPETITION}, "
                f"{ATLAS_TARGET_FORCE_N:g} N"
            )
        frame = min(
            candidates,
            key=lambda candidate: abs(_force(candidate) - ATLAS_TARGET_FORCE_N),
        )
        selected.append(
            AtlasSelection(
                material=condition.material,
                morphology=condition.morphology,
                display_name=condition.display_name,
                displayed_contact_position_mm=HOLE_TO_CONTACT_X_MM[hole],
                hole_index=hole,
                repetition_index=ATLAS_REPETITION,
                target_force_n=ATLAS_TARGET_FORCE_N,
                actual_force_n=_force(frame),
                image_path=frame.rgb_path.resolve(),
                timestamp_s=_time(frame),
                unloaded=False,
                run_id=frame.run.run_id,
                capture_id="",
            )
        )
    return selected


def _select_unloaded_frame(
    condition: object, loaded: list[AtlasSelection]
) -> AtlasSelection:
    assert condition.session_path is not None
    index = index_session(condition.session_path, expected_repetitions=5)
    unloaded = [frame for frame in index.frames if frame.run is None]
    if not unloaded:
        raise RuntimeError(f"missing unloaded capture for {condition.display_name}")
    reference_time = float(np.median([selection.timestamp_s for selection in loaded]))
    frame = min(unloaded, key=lambda candidate: abs(_time(candidate) - reference_time))
    return AtlasSelection(
        material=condition.material,
        morphology=condition.morphology,
        display_name=condition.display_name,
        displayed_contact_position_mm=None,
        hole_index=None,
        repetition_index=None,
        target_force_n=0.0,
        actual_force_n=_force(frame),
        image_path=frame.rgb_path.resolve(),
        timestamp_s=_time(frame),
        unloaded=True,
        run_id="",
        capture_id=frame.segment_path.name,
    )


def select_atlas_frames() -> dict[str, tuple[AtlasSelection, ...] | None]:
    """Return six ordered row selections, preserving the pending condition."""

    require_available_inputs()
    result: dict[str, tuple[AtlasSelection, ...] | None] = {}
    for condition in MORPHOLOGY_CONDITIONS:
        if condition.pending:
            result[condition.display_name] = None
            continue
        loaded = _select_loaded_frames(condition)
        unloaded = _select_unloaded_frame(condition, loaded)
        result[condition.display_name] = tuple([unloaded, *loaded])
    return result


def write_selection_manifest(
    selections: dict[str, tuple[AtlasSelection, ...] | None],
    path: Path = FIGURE_DIRECTORY / "fig5a_selection_manifest.csv",
) -> Path:
    """Persist exactly which raw frames entered Figure 5(a)."""

    fields = (
        "material",
        "morphology",
        "display_name",
        "displayed_contact_position_mm",
        "hole_index",
        "repetition_index",
        "target_force_n",
        "actual_force_n",
        "image_path",
        "timestamp_s",
        "unloaded",
        "run_id",
        "capture_id",
        "display_exposure_ev",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selections.values():
            if row is None:
                continue
            for item in row:
                record = {
                    field: getattr(item, field)
                    for field in fields
                    if field not in {"image_path", "display_exposure_ev"}
                }
                record["display_exposure_ev"] = (
                    ATLAS_DISPLAY_EXPOSURE_EV_BY_MATERIAL[item.material]
                )
                try:
                    record["image_path"] = str(item.image_path.relative_to(REPOSITORY_ROOT))
                except ValueError:
                    record["image_path"] = str(item.image_path)
                writer.writerow(record)
    return path


def _crop(rgb: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = ATLAS_CROP_XYXY
    if rgb.shape[1] < x1 or rgb.shape[0] < y1:
        raise ValueError(
            f"atlas ROI {ATLAS_CROP_XYXY} exceeds image shape {rgb.shape[:2]}"
        )
    return rgb[y0:y1, x0:x1]


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    panel_label: str = "(a)",
    selections: dict[str, tuple[AtlasSelection, ...] | None] | None = None,
) -> dict[str, tuple[AtlasSelection, ...] | None]:
    """Render the atlas into one supplied SubplotSpec."""

    if selections is None:
        selections = select_atlas_frames()
    write_selection_manifest(selections)

    grid = subplot_spec.subgridspec(
        9,
        7,
        height_ratios=MORPHOLOGY_TABLE_HEIGHT_RATIOS,
        width_ratios=(1.18, 1, 1, 1, 1, 1, 1),
        hspace=MORPHOLOGY_TABLE_HSPACE,
        wspace=0.010,
    )
    title_axis = figure.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.55,
        panel_label,
        fontsize=DEFAULT_STYLE.panel_label_font_size_pt,
        fontweight="bold",
        va="center",
    )
    title_axis.text(
        0.080,
        0.55,
        "Measured optical signatures across contact locations",
        fontsize=6.2,
        fontweight="bold",
        va="center",
    )

    headers = ["Zero\nload"] + [
        "$X_{\\mathrm{contact}}$\n"
        f"{HOLE_TO_CONTACT_X_MM[hole]:g} mm"
        for hole in ATLAS_HOLES
    ]
    for column, header in enumerate(headers, start=1):
        axis = figure.add_subplot(grid[1, column])
        axis.axis("off")
        axis.text(0.5, 0.52, header, fontsize=5.0, ha="center", va="center")

    for condition, row_slot in zip(
        MORPHOLOGY_CONDITIONS, MORPHOLOGY_TABLE_ROW_SLOTS, strict=True
    ):
        label_axis = figure.add_subplot(grid[row_slot, 0])
        label_axis.axis("off")
        material, morphology = condition.display_name.rsplit(" ", 1)
        label_axis.text(
            0.0,
            0.5,
            f"{material}\n{morphology}",
            fontsize=4.9,
            ha="left",
            va="center",
            linespacing=1.08,
        )
        row = selections[condition.display_name]
        for column in range(6):
            axis = figure.add_subplot(grid[row_slot, column + 1])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#D2D2D2")
                spine.set_linewidth(0.32)
            if row is None:
                x0, y0, x1, y1 = ATLAS_CROP_XYXY
                placeholder = np.full((y1 - y0, x1 - x0, 3), 241, dtype=np.uint8)
                axis.imshow(placeholder)
                axis.text(
                    0.5,
                    0.5,
                    "pending",
                    color="#777777",
                    fontsize=5.3,
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            else:
                rgb = _crop(load_rgb(row[column].image_path))
                exposure_ev = ATLAS_DISPLAY_EXPOSURE_EV_BY_MATERIAL[
                    condition.material
                ]
                if exposure_ev != 0.0:
                    rgb = np.clip(
                        rgb.astype(np.float32) * (2.0**exposure_ev), 0.0, 255.0
                    ).astype(np.uint8)
                axis.imshow(rgb)
    return selections


def main() -> None:
    """Export a standalone debug render of Figure 5(a)."""

    with publication_context(DEFAULT_STYLE):
        figure = plt.figure(figsize=(7.16, 4.25))
        grid = figure.add_gridspec(1, 1, left=0.015, right=0.995, bottom=0.015, top=0.99)
        render_panel(figure, grid[0, 0])
        save_figure(
            figure,
            FIGURE_DIRECTORY / "fig5a",
            formats=("png",),
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
