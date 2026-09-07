"""Panel-rendering tools for the Figure 5(b) response-field heatmaps."""

from __future__ import annotations

import csv

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from lumo.visualization import DEFAULT_STYLE, MATERIAL_LABELS, PAPER_LABELS

from .config import (
    ALL_HOLES,
    ANALYSIS_CONDITION_OVERRIDES,
    ANALYSIS_ROOTS,
    COMPARISON_CONDITIONS,
    COMPARISON_MORPHOLOGIES,
    FIGURE_DIRECTORY,
    HOLE_TO_CONTACT_X_MM,
    MATERIAL_SEPARATOR_COLOR,
    MATERIAL_SEPARATOR_LINEWIDTH_PT,
    MORPHOLOGY_TABLE_HEIGHT_RATIOS,
    MORPHOLOGY_TABLE_HSPACE,
    MORPHOLOGY_TABLE_ROW_SLOTS,
    require_available_inputs,
)


INDENTER_COLUMNS = (
    ("sphere_10mm", "10 mm sphere"),
    ("sphere_30mm", "30 mm sphere"),
)
PLOT_COLUMNS = (2, 3, 5, 6)
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
    show_row_labels: bool = True,
    data: tuple[
        np.ndarray,
        np.ndarray,
        dict[tuple[str, str, str], np.ndarray | None],
        list[dict[str, object]],
    ]
    | None = None,
) -> dict[str, object]:
    """Render shared-scale regional 2-to-15 N optical changes."""

    coordinate, region_edges, responses, audit_rows = (
        load_optical_change_maps() if data is None else data
    )
    write_region_response_csv(audit_rows)
    measured = [values for values in responses.values() if values is not None]
    maximum = max(float(np.max(values)) for values in measured)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("coarse response magnitudes must have a finite positive range")
    normalization = Normalize(vmin=0.0, vmax=maximum)

    row_label_width = 0.13 if show_row_labels else 0.012
    grid = subplot_spec.subgridspec(
        6,
        8,
        height_ratios=MORPHOLOGY_TABLE_HEIGHT_RATIOS,
        width_ratios=(0.10, row_label_width, 1, 1, 0.04, 1, 1, 0.045),
        hspace=MORPHOLOGY_TABLE_HSPACE,
        wspace=0.02,
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
        0.140,
        0.55,
        "Measured optical change from 2 to 15 N",
        fontsize=6.2,
        fontweight="bold",
        va="center",
    )

    for material, column_slice in (
        ("solaris", slice(2, 4)),
        ("dragon_skin", slice(5, 7)),
    ):
        material_axis = figure.add_subplot(grid[1, column_slice])
        material_axis.axis("off")
        material_axis.text(
            0.5,
            0.55,
            MATERIAL_LABELS[material],
            fontsize=4.8,
            fontweight="bold",
            ha="center",
            va="center",
        )

    indenter_titles = dict(INDENTER_COLUMNS)
    for column, (_, candidate_indenter, _) in zip(
        PLOT_COLUMNS, COMPARISON_CONDITIONS, strict=True
    ):
        column_axis = figure.add_subplot(grid[2, column])
        column_axis.axis("off")
        column_axis.text(
            0.5,
            0.52,
            indenter_titles[candidate_indenter],
            fontsize=4.5,
            ha="center",
            va="center",
        )

    image = None
    axes = []
    physical_locations = np.asarray(
        [HOLE_TO_CONTACT_X_MM[hole] for hole in ALL_HOLES], dtype=np.float64
    )
    shared_y_axis = figure.add_subplot(grid[3:, 0])
    shared_y_axis.axis("off")
    shared_y_axis.text(
        -0.70,
        0.5,
        r"$X_{\mathrm{contact}}$ [mm]",
        fontsize=3.9,
        rotation=90,
        ha="center",
        va="center",
    )

    separator_axis = figure.add_subplot(grid[1:, 4])
    separator_axis.axis("off")
    separator_axis.plot(
        (0.5, 0.5),
        (0.0, 1.0),
        color=MATERIAL_SEPARATOR_COLOR,
        linewidth=MATERIAL_SEPARATOR_LINEWIDTH_PT,
        transform=separator_axis.transAxes,
        clip_on=False,
    )

    for row_index, (morphology, row_slot) in enumerate(
        zip(COMPARISON_MORPHOLOGIES, MORPHOLOGY_TABLE_ROW_SLOTS, strict=True)
    ):
        if show_row_labels:
            row_label_axis = figure.add_subplot(grid[row_slot, 1])
            row_label_axis.axis("off")
            row_label_axis.text(
                0.05,
                0.5,
                PAPER_LABELS[morphology],
                fontsize=3.7,
                ha="left",
                va="center",
            )
        for plot_index, ((material, candidate_indenter, _), column) in enumerate(
            zip(COMPARISON_CONDITIONS, PLOT_COLUMNS, strict=True)
        ):
            axis = figure.add_subplot(grid[row_slot, column])
            key = (material, candidate_indenter, morphology)
            values = responses[key]
            if values is None:
                axis.set_facecolor("#EFEFEF")
                axis.set_xlim(-0.5, N_LONGITUDINAL_REGIONS - 0.5)
                axis.set_ylim(60.5, -5.5)
                axis.text(
                    0.5,
                    0.5,
                    "pending",
                    transform=axis.transAxes,
                    fontsize=5.5,
                    color="#888888",
                    ha="center",
                    va="center",
                )
            else:
                image = axis.imshow(
                    values,
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                    norm=normalization,
                    extent=(-0.5, N_LONGITUDINAL_REGIONS - 0.5, 60.5, -5.5),
                )
                peak_regions = np.argmax(values, axis=1)
                markers = axis.scatter(
                    peak_regions,
                    physical_locations,
                    marker="x",
                    s=8.0,
                    linewidths=0.70,
                    color="white",
                    zorder=3,
                )
                markers.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=1.25, foreground="#333333"),
                        path_effects.Normal(),
                    ]
                )
            if plot_index == 0:
                axis.set_yticks(physical_locations)
            else:
                axis.set_yticks([])
            if (
                row_index == len(COMPARISON_MORPHOLOGIES) - 1
                and plot_index in (0, 2)
            ):
                axis.set_xticks(
                    np.arange(N_LONGITUDINAL_REGIONS),
                    tuple(
                        f"R{region}" for region in range(1, N_LONGITUDINAL_REGIONS + 1)
                    ),
                )
                if plot_index == 0:
                    axis.text(
                        0.0,
                        -0.15,
                        "Distal",
                        transform=axis.transAxes,
                        fontsize=4.4,
                        ha="left",
                        va="top",
                    )
            else:
                axis.set_xticks([])
            axis.tick_params(labelsize=3.8, length=1.3, pad=0.6)
            axis.set_box_aspect(1.0)
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#777777")
            axes.append(axis)

    assert image is not None
    colorbar_axis = figure.add_subplot(grid[3:, 7])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.ax.set_title(
        "$\\Delta$ signal\n[DN]",
        fontsize=3.1,
        pad=1.0,
        x=1.0,
        ha="right",
    )
    colorbar.ax.tick_params(labelsize=4.5, length=1.5, pad=0.8)
    colorbar.outline.set_linewidth(0.45)
    body_position = grid[3:, 2:7].get_position(figure)
    figure.text(
        body_position.x1,
        body_position.y0 - 0.014,
        "Proximal",
        fontsize=4.4,
        ha="right",
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
