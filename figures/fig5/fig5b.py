"""Panel-rendering tools for Figure 5(b) spatial optical signal maps."""

from __future__ import annotations

import csv

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from lumo.visualization import DEFAULT_STYLE

from .config import (
    ALL_HOLES,
    ANALYSIS_CONDITION_OVERRIDES,
    ANALYSIS_ROOTS,
    COMPARISON_CONDITIONS,
    COMPARISON_MORPHOLOGIES,
    FIGURE_DIRECTORY,
    HOLE_TO_CONTACT_X_MM,
    require_available_inputs,
)
from .table_layout import (
    HEADER_ROW,
    MATERIAL_BLOCKS,
    TABLE_HEIGHT_RATIOS,
    TABLE_HSPACE,
    add_panel_title,
)


INDENTER_COLUMNS = (
    ("sphere_10mm", "Ø10 mm\nsphere"),
    ("sphere_30mm", "Ø30 mm\nsphere"),
)
PLOT_COLUMNS = (2, 3)
N_LONGITUDINAL_REGIONS = 6
FORCE_LOW_N = 2.0
FORCE_HIGH_N = 15.0


def _fixed_region_indices(
    coordinate: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return six fixed regions over the normalized longitudinal coordinate."""

    if coordinate.ndim != 1 or len(coordinate) < N_LONGITUDINAL_REGIONS:
        raise ValueError("longitudinal coordinate cannot define six regions")
    if not np.all(np.isfinite(coordinate)) or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("longitudinal coordinate must be finite and increasing")
    if not np.isclose(coordinate[0], 0.0) or not np.isclose(coordinate[-1], 1.0):
        raise ValueError("Figure 5(b) expects a normalized 0-to-1 coordinate")

    edges = np.linspace(0.0, 1.0, N_LONGITUDINAL_REGIONS + 1)
    regions = []
    for region in range(N_LONGITUDINAL_REGIONS):
        if region == N_LONGITUDINAL_REGIONS - 1:
            mask = (coordinate >= edges[region]) & (coordinate <= edges[region + 1])
        else:
            mask = (coordinate >= edges[region]) & (coordinate < edges[region + 1])
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError(f"longitudinal region {region + 1} contains no samples")
        regions.append(indices)
    return edges, tuple(regions)


def load_optical_change_maps() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[tuple[str, str, str], np.ndarray | None],
    list[dict[str, object]],
]:
    """Load paired 2/15 N profiles and form median regional RMS changes."""

    require_available_inputs()
    coordinate: np.ndarray | None = None
    region_edges: np.ndarray | None = None
    region_indices: tuple[np.ndarray, ...] | None = None
    response_maps: dict[tuple[str, str, str], np.ndarray | None] = {}
    audit_rows: list[dict[str, object]] = []
    for candidate_material, root in ANALYSIS_ROOTS.items():
        material_indenters = [
            candidate_indenter
            for row_material, candidate_indenter, _ in COMPARISON_CONDITIONS
            if row_material == candidate_material
        ]
        for candidate_indenter in material_indenters:
            for candidate_morphology in COMPARISON_MORPHOLOGIES:
                key = (
                    candidate_material,
                    candidate_indenter,
                    candidate_morphology,
                )
                source_root = ANALYSIS_CONDITION_OVERRIDES.get(
                    (candidate_material, candidate_morphology, candidate_indenter),
                    root,
                )
                path = source_root / "raw_data_summary" / "longitudinal_profiles.npz"
                with np.load(path, allow_pickle=False) as data:
                    current_coordinate = np.asarray(
                        data["longitudinal_coordinate"], dtype=np.float64
                    )
                    profiles = np.asarray(data["profiles"], dtype=np.float64)
                    specimen = np.asarray(data["specimen_id"]).astype(str)
                    material = np.asarray(data["material"]).astype(str)
                    morphology = np.asarray(data["morphology"]).astype(str)
                    run_id = np.asarray(data["run_id"]).astype(str)
                    indenter = np.asarray(data["indenter"]).astype(str)
                    holes = np.asarray(data["hole_index"], dtype=np.int64)
                    repetition = np.asarray(data["repetition_index"], dtype=np.int64)
                    target_force = np.asarray(data["target_force_n"], dtype=np.float64)
                    actual_force = np.asarray(data["actual_force_n"], dtype=np.float64)
                    status = np.asarray(data["run_status"]).astype(str)

                if coordinate is None:
                    coordinate = current_coordinate
                    region_edges, region_indices = _fixed_region_indices(coordinate)
                elif not np.array_equal(coordinate, current_coordinate):
                    raise RuntimeError(
                        "Figure 5(b) summaries use different optical coordinates"
                    )
                assert region_edges is not None and region_indices is not None

                contact_rows = []
                for hole in ALL_HOLES:
                    condition_mask = (
                        (material == candidate_material)
                        & (morphology == candidate_morphology)
                        & (indenter == candidate_indenter)
                        & (holes == hole)
                        & (status == "complete")
                    )
                    repetitions = np.unique(repetition[condition_mask])
                    if len(repetitions) != 5:
                        raise RuntimeError(
                            "Figure 5(b) requires five independent repetitions for "
                            f"{candidate_material}, {candidate_morphology}, "
                            f"{candidate_indenter}, hole {hole}; found {len(repetitions)}"
                        )
                    repetition_values = []
                    repetition_metadata = []
                    for candidate_repetition in repetitions:
                        repetition_mask = condition_mask & (
                            repetition == candidate_repetition
                        )
                        low_indices = np.flatnonzero(
                            repetition_mask & np.isclose(target_force, FORCE_LOW_N)
                        )
                        high_indices = np.flatnonzero(
                            repetition_mask & np.isclose(target_force, FORCE_HIGH_N)
                        )
                        if len(low_indices) != 1 or len(high_indices) != 1:
                            raise RuntimeError(
                                "Figure 5(b) requires exactly one 2 N and one 15 N "
                                "profile within each complete repetition; "
                                f"{key}, hole {hole}, repetition "
                                f"{candidate_repetition} has {len(low_indices)} / "
                                f"{len(high_indices)}"
                            )
                        low_index = int(low_indices[0])
                        high_index = int(high_indices[0])
                        if run_id[low_index] != run_id[high_index]:
                            raise RuntimeError(
                                "Figure 5(b) low/high profiles do not share a run: "
                                f"{key}, hole {hole}, repetition {candidate_repetition}"
                            )
                        delta = profiles[high_index] - profiles[low_index]
                        regional_rms = np.asarray(
                            [
                                np.sqrt(np.mean(np.square(delta[indices])))
                                for indices in region_indices
                            ],
                            dtype=np.float64,
                        )
                        repetition_values.append(regional_rms)
                        repetition_metadata.append(
                            (
                                int(candidate_repetition),
                                specimen[low_index],
                                run_id[low_index],
                                actual_force[low_index],
                                actual_force[high_index],
                            )
                        )

                    repetition_values_array = np.asarray(repetition_values)
                    median_values = np.median(repetition_values_array, axis=0)
                    peak_region = int(np.argmax(median_values))
                    contact_rows.append(median_values)
                    for repetition_row, metadata in enumerate(repetition_metadata):
                        (
                            candidate_repetition,
                            candidate_specimen,
                            candidate_run_id,
                            actual_low,
                            actual_high,
                        ) = metadata
                        for region in range(N_LONGITUDINAL_REGIONS):
                            audit_rows.append(
                                {
                                    "specimen_id": candidate_specimen,
                                    "material": candidate_material,
                                    "morphology": candidate_morphology,
                                    "run_id": candidate_run_id,
                                    "indenter": candidate_indenter,
                                    "X_contact_mm": HOLE_TO_CONTACT_X_MM[hole],
                                    "repetition_index": candidate_repetition,
                                    "force_low_target_n": FORCE_LOW_N,
                                    "force_high_target_n": FORCE_HIGH_N,
                                    "force_low_actual_n": actual_low,
                                    "force_high_actual_n": actual_high,
                                    "region_index": region + 1,
                                    "region_start_normalized": region_edges[region],
                                    "region_end_normalized": region_edges[region + 1],
                                    "delta_rms_DN": repetition_values_array[
                                        repetition_row, region
                                    ],
                                    "median_delta_rms_DN": median_values[region],
                                    "is_peak_region": region == peak_region,
                                }
                            )
                response_maps[key] = np.asarray(contact_rows)

    if coordinate is None or region_edges is None:
        raise RuntimeError("Figure 5(b) has no longitudinal-profile summaries")
    return coordinate, region_edges, response_maps, audit_rows


def write_region_response_csv(
    rows: list[dict[str, object]],
) -> None:
    """Persist each paired repetition and the medians displayed in (b)."""

    path = FIGURE_DIRECTORY / "fig5b_region_response.csv"
    fields = (
        "specimen_id",
        "material",
        "morphology",
        "run_id",
        "indenter",
        "X_contact_mm",
        "repetition_index",
        "force_low_target_n",
        "force_high_target_n",
        "force_low_actual_n",
        "force_high_actual_n",
        "region_index",
        "region_start_normalized",
        "region_end_normalized",
        "delta_rms_DN",
        "median_delta_rms_DN",
        "is_peak_region",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    panel_label: str = "(b)",
    data: tuple[
        np.ndarray,
        np.ndarray,
        dict[tuple[str, str, str], np.ndarray | None],
        list[dict[str, object]],
    ]
    | None = None,
) -> dict[str, object]:
    """Render shared-scale spatial optical signal maps."""

    coordinate, region_edges, responses, audit_rows = (
        load_optical_change_maps() if data is None else data
    )
    write_region_response_csv(audit_rows)
    measured = [values for values in responses.values() if values is not None]
    maximum = max(float(np.max(values)) for values in measured)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("response magnitudes must have a finite positive range")
    normalization = Normalize(vmin=0.0, vmax=maximum)

    grid = subplot_spec.subgridspec(
        len(TABLE_HEIGHT_RATIOS),
        4,
        height_ratios=TABLE_HEIGHT_RATIOS,
        width_ratios=(0.20, 0.20, 1, 1),
        hspace=TABLE_HSPACE,
        wspace=0.07,
    )
    add_panel_title(
        figure,
        grid,
        panel_label=panel_label,
        title="Spatial optical response, 2–15 N",
        sample_note="Median of n = 5 re-contacts / location",
    )

    indenter_titles = dict(INDENTER_COLUMNS)
    for column, (candidate_indenter, _) in zip(
        PLOT_COLUMNS, INDENTER_COLUMNS, strict=True
    ):
        column_axis = figure.add_subplot(grid[HEADER_ROW, column])
        column_axis.axis("off")
        column_axis.text(
            0.5,
            0.76,
            indenter_titles[candidate_indenter],
            fontsize=DEFAULT_STYLE.condition_header_font_size_pt,
            linespacing=0.90,
            ha="center",
            va="center",
        )

    axes = []
    shared_y_axis = figure.add_subplot(grid[2:, 0])
    shared_y_axis.axis("off")
    shared_y_axis.text(
        0.5,
        0.5,
        "Longitudinal region",
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt,
        rotation=90,
        ha="center",
        va="center",
        transform=shared_y_axis.transAxes,
        clip_on=False,
        zorder=2,
    )

    image = None
    for material, morphology_rows in MATERIAL_BLOCKS:
        for morphology_index, (morphology, row_slot) in enumerate(
            zip(COMPARISON_MORPHOLOGIES, morphology_rows, strict=True)
        ):
            for plot_index, ((candidate_indenter, _), column) in enumerate(
                zip(INDENTER_COLUMNS, PLOT_COLUMNS, strict=True)
            ):
                axis = figure.add_subplot(grid[row_slot, column])
                key = (material, candidate_indenter, morphology)
                values = responses[key]
                if values is None:
                    axis.set_facecolor("#EFEFEF")
                    axis.set_xlim(-0.5, N_LONGITUDINAL_REGIONS - 0.5)
                    axis.set_ylim(N_LONGITUDINAL_REGIONS - 0.5, -0.5)
                    axis.text(
                        0.5,
                        0.5,
                        "pending",
                        transform=axis.transAxes,
                        fontsize=DEFAULT_STYLE.annotation_font_size_pt,
                        color="#888888",
                        ha="center",
                        va="center",
                    )
                else:
                    image = axis.imshow(
                        values.T,
                        aspect="auto",
                        interpolation="nearest",
                        cmap="viridis",
                        norm=normalization,
                        origin="upper",
                    )
                axis.set_xlim(-0.5, N_LONGITUDINAL_REGIONS - 0.5)
                axis.set_ylim(N_LONGITUDINAL_REGIONS - 0.5, -0.5)
                axis.set_yticks((0, 2, 5))
                axis.set_yticklabels(
                    ("R1", "R3", "R6") if plot_index == 0 else ()
                )
                if (
                    material == "dragon_skin"
                    and morphology_index == len(COMPARISON_MORPHOLOGIES) - 1
                ):
                    axis.set_xticks(
                        (0, 2, 4),
                        ("0", "20", "40"),
                    )
                else:
                    axis.set_xticks([])
                axis.tick_params(
                    labelsize=DEFAULT_STYLE.minimum_font_size_pt,
                    width=DEFAULT_STYLE.tick_width_pt,
                    length=DEFAULT_STYLE.tick_length_pt,
                    pad=0.8,
                    left=(plot_index == 0),
                    labelleft=(plot_index == 0),
                )
                for spine in axis.spines.values():
                    spine.set_linewidth(DEFAULT_STYLE.spine_width_pt)
                    spine.set_color(DEFAULT_STYLE.colors.neutral)
                axes.append(axis)

    if image is None:
        raise ValueError("no Figure 5(b) response heatmaps were rendered")

    colorbar_container = figure.add_subplot(grid[HEADER_ROW, 0:4])
    colorbar_container.axis("off")
    colorbar_container.text(
        0.50,
        0.50,
        "Optical change [DN]",
        fontsize=DEFAULT_STYLE.axis_label_font_size_pt,
        ha="center",
        va="center",
    )
    colorbar_axis = colorbar_container.inset_axes((0.28, 0.24, 0.44, 0.06))
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks((0.0, maximum))
    colorbar.ax.set_xticklabels(("0", f"{maximum:.1f}"))
    colorbar.ax.xaxis.set_ticks_position("bottom")
    colorbar.ax.tick_params(
        labelsize=DEFAULT_STYLE.minimum_font_size_pt,
        width=DEFAULT_STYLE.tick_width_pt,
        length=DEFAULT_STYLE.tick_length_pt,
        pad=0.35,
    )
    colorbar.outline.set_linewidth(DEFAULT_STYLE.spine_width_pt)

    body_position = grid[2:, 2:4].get_position(figure)
    figure.text(
        0.5 * (body_position.x0 + body_position.x1),
        body_position.y0 - 0.045,
        "Contact location [mm]",
        fontsize=DEFAULT_STYLE.annotation_font_size_pt,
        ha="center",
        va="top",
    )
    return {
        "axes": tuple(axes),
        "coordinate": coordinate,
        "region_edges": region_edges,
        "region_responses": responses,
        "audit_rows": audit_rows,
        "color_limits": (0.0, maximum),
    }
