"""Figure 5(b): measured hardware response-field heatmaps."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import SubplotSpec  # noqa: E402

from lumo.visualization import DEFAULT_STYLE, publication_context, save_figure  # noqa: E402

from .config import (  # noqa: E402
    ALL_HOLES,
    ANALYSIS_ROOTS,
    COMPARISON_CONDITIONS,
    COMPARISON_MORPHOLOGIES,
    FIGURE_DIRECTORY,
    HOLE_TO_CONTACT_X_MM,
    MORPHOLOGY_CONDITIONS,
    MORPHOLOGY_TABLE_HEIGHT_RATIOS,
    MORPHOLOGY_TABLE_HSPACE,
    MORPHOLOGY_TABLE_ROW_SLOTS,
    require_available_inputs,
)


INDENTER_COLUMNS = (
    ("sphere_10mm", "10 mm sphere"),
    ("sphere_30mm", "30 mm sphere"),
)
N_LONGITUDINAL_REGIONS = 6


def load_response_templates() -> tuple[
    np.ndarray, dict[tuple[str, str, str], np.ndarray | None]
]:
    """Load compact slopes and median independent repetitions per location."""

    require_available_inputs()
    coordinate: np.ndarray | None = None
    templates: dict[tuple[str, str, str], np.ndarray | None] = {}
    for candidate_material, root in ANALYSIS_ROOTS.items():
        path = root / "raw_data_summary" / "load_response_profiles.npz"
        with np.load(path, allow_pickle=False) as data:
            current_coordinate = np.asarray(
                data["longitudinal_coordinate"], dtype=np.float64
            )
            profiles = np.asarray(data["slope_profiles"], dtype=np.float64)
            material = np.asarray(data["material"]).astype(str)
            morphology = np.asarray(data["morphology"]).astype(str)
            indenter = np.asarray(data["indenter"]).astype(str)
            holes = np.asarray(data["hole_index"], dtype=np.int64)
            repetition = np.asarray(data["repetition_index"], dtype=np.int64)
            status = np.asarray(data["run_status"]).astype(str)

        if coordinate is None:
            coordinate = current_coordinate
        elif not np.array_equal(coordinate, current_coordinate):
            raise RuntimeError("Figure 5(b) summaries use different optical coordinates")

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
                if (
                    candidate_material == "dragon_skin"
                    and candidate_morphology == "angled_opt"
                ):
                    templates[key] = None
                    continue
                rows = []
                for hole in ALL_HOLES:
                    mask = (
                        (material == candidate_material)
                        & (morphology == candidate_morphology)
                        & (indenter == candidate_indenter)
                        & (holes == hole)
                        & (status == "complete")
                    )
                    indices = np.flatnonzero(mask)
                    if len(indices) != 5 or len(np.unique(repetition[indices])) != 5:
                        raise RuntimeError(
                            "Figure 5(b) requires five independent repetitions for "
                            f"{candidate_material}, {candidate_morphology}, "
                            f"{candidate_indenter}, hole {hole}; found {len(indices)}"
                        )
                    rows.append(np.median(profiles[indices], axis=0))
                templates[key] = np.asarray(rows)

    if coordinate is None:
        raise RuntimeError("Figure 5(b) has no response-profile summaries")
    return coordinate, templates


def coarse_region_responses(
    coordinate: np.ndarray,
    templates: dict[tuple[str, str, str], np.ndarray | None],
) -> tuple[np.ndarray, dict[tuple[str, str, str], np.ndarray | None]]:
    """Reduce each dense signed profile to six fixed-region RMS magnitudes."""

    if coordinate.ndim != 1 or len(coordinate) < N_LONGITUDINAL_REGIONS:
        raise ValueError("longitudinal coordinate cannot define six regions")
    if not np.all(np.isfinite(coordinate)) or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("longitudinal coordinate must be finite and increasing")

    edges = np.linspace(
        float(coordinate[0]),
        float(coordinate[-1]),
        N_LONGITUDINAL_REGIONS + 1,
    )
    region_indices = []
    for region in range(N_LONGITUDINAL_REGIONS):
        if region == N_LONGITUDINAL_REGIONS - 1:
            mask = (coordinate >= edges[region]) & (coordinate <= edges[region + 1])
        else:
            mask = (coordinate >= edges[region]) & (coordinate < edges[region + 1])
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError(f"longitudinal region {region + 1} contains no samples")
        region_indices.append(indices)

    coarse: dict[tuple[str, str, str], np.ndarray | None] = {}
    for key, values in templates.items():
        if values is None:
            coarse[key] = None
            continue
        if values.ndim != 2 or values.shape[1] != coordinate.size:
            raise ValueError(f"response template has incompatible shape for {key}")
        coarse[key] = np.column_stack(
            [
                np.sqrt(np.mean(np.square(values[:, indices]), axis=1))
                for indices in region_indices
            ]
        )
    return edges, coarse


def write_region_response_csv(
    region_edges: np.ndarray,
    responses: dict[tuple[str, str, str], np.ndarray | None],
) -> None:
    """Persist the exact coarse values and dominant-region flags shown in (b)."""

    path = FIGURE_DIRECTORY / "fig5b_region_response.csv"
    fields = (
        "material",
        "morphology",
        "indenter",
        "X_contact_mm",
        "region_index",
        "region_start",
        "region_end",
        "response_rms_DN_per_N",
        "is_peak_region",
    )
    physical_locations = [HOLE_TO_CONTACT_X_MM[hole] for hole in ALL_HOLES]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (material, indenter, morphology), values in responses.items():
            if values is None:
                continue
            peaks = np.argmax(values, axis=1)
            for contact_row, contact_position in enumerate(physical_locations):
                for region in range(N_LONGITUDINAL_REGIONS):
                    writer.writerow(
                        {
                            "material": material,
                            "morphology": morphology,
                            "indenter": indenter,
                            "X_contact_mm": contact_position,
                            "region_index": region + 1,
                            "region_start": region_edges[region],
                            "region_end": region_edges[region + 1],
                            "response_rms_DN_per_N": values[contact_row, region],
                            "is_peak_region": region == peaks[contact_row],
                        }
                    )


def render_panel(
    figure: Figure,
    subplot_spec: SubplotSpec,
    *,
    panel_label: str = "(b)",
    data: tuple[
        np.ndarray, dict[tuple[str, str, str], np.ndarray | None]
    ]
    | None = None,
) -> dict[str, object]:
    """Render the shared-scale response fields into one SubplotSpec."""

    coordinate, templates = load_response_templates() if data is None else data
    region_edges, responses = coarse_region_responses(coordinate, templates)
    write_region_response_csv(region_edges, responses)
    measured = [values for values in responses.values() if values is not None]
    maximum = max(float(np.max(values)) for values in measured)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("coarse response magnitudes must have a finite positive range")
    normalization = Normalize(vmin=0.0, vmax=maximum)

    grid = subplot_spec.subgridspec(
        9,
        6,
        height_ratios=MORPHOLOGY_TABLE_HEIGHT_RATIOS,
        width_ratios=(0.10, 0.54, 1.0, 1.0, 0.055, 0.13),
        hspace=MORPHOLOGY_TABLE_HSPACE,
        wspace=0.045,
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
        "Measured hardware response fields",
        fontsize=6.2,
        fontweight="bold",
        va="center",
    )

    for column, (_, column_title) in enumerate(INDENTER_COLUMNS, start=2):
        column_axis = figure.add_subplot(grid[1, column])
        column_axis.axis("off")
        column_axis.text(
            0.5,
            0.52,
            column_title,
            fontsize=5.2,
            ha="center",
            va="center",
        )

    shared_y_axis = figure.add_subplot(grid[2:, 0])
    shared_y_axis.axis("off")
    shared_y_axis.text(
        0.42,
        0.5,
        r"$X_{\mathrm{contact}}$ [mm]",
        rotation=90,
        fontsize=5.1,
        ha="center",
        va="center",
    )

    image = None
    axes = []
    physical_locations = np.asarray(
        [HOLE_TO_CONTACT_X_MM[hole] for hole in ALL_HOLES], dtype=np.float64
    )
    for row_index, (condition, row_slot) in enumerate(
        zip(MORPHOLOGY_CONDITIONS, MORPHOLOGY_TABLE_ROW_SLOTS, strict=True)
    ):
        row_label_axis = figure.add_subplot(grid[row_slot, 1])
        row_label_axis.axis("off")
        material_name, morphology_name = condition.display_name.rsplit(" ", 1)
        row_label_axis.text(
            0.0,
            0.5,
            f"{material_name}\n{morphology_name}",
            fontsize=4.9,
            ha="left",
            va="center",
            linespacing=1.08,
        )
        for column, (candidate_indenter, _) in enumerate(INDENTER_COLUMNS, start=2):
            axis = figure.add_subplot(grid[row_slot, column])
            key = (condition.material, candidate_indenter, condition.morphology)
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
            if column == 2 and row_index in (0, 3):
                axis.set_yticks(physical_locations)
            else:
                axis.set_yticks([])
            if row_index == len(MORPHOLOGY_CONDITIONS) - 1:
                axis.set_xticks(
                    np.arange(N_LONGITUDINAL_REGIONS),
                    tuple(f"R{region}" for region in range(1, N_LONGITUDINAL_REGIONS + 1)),
                )
                if column == 2:
                    axis.text(
                        0.0,
                        -0.25,
                        "Distal",
                        transform=axis.transAxes,
                        fontsize=4.4,
                        ha="left",
                        va="top",
                    )
                else:
                    axis.text(
                        1.0,
                        -0.25,
                        "Proximal",
                        transform=axis.transAxes,
                        fontsize=4.4,
                        ha="right",
                        va="top",
                    )
            else:
                axis.set_xticks([])
            axis.tick_params(labelsize=4.6, length=1.5, pad=0.8)
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#777777")
            axes.append(axis)

    assert image is not None
    colorbar_axis = figure.add_subplot(grid[2:, 4])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.ax.set_title("Response\nmagnitude\n[DN/N]", fontsize=3.9, pad=1.5)
    colorbar.ax.tick_params(labelsize=4.5, length=1.5, pad=0.8)
    colorbar.outline.set_linewidth(0.45)
    return {
        "axes": tuple(axes),
        "templates": templates,
        "coordinate": coordinate,
        "region_edges": region_edges,
        "region_responses": responses,
        "color_limits": (0.0, maximum),
    }


def main() -> None:
    """Export a standalone debug render of Figure 5(b)."""

    with publication_context(DEFAULT_STYLE):
        figure = plt.figure(figsize=(7.16, 4.25))
        grid = figure.add_gridspec(1, 1, left=0.02, right=0.985, bottom=0.075, top=0.99)
        render_panel(figure, grid[0, 0])
        save_figure(
            figure,
            FIGURE_DIRECTORY / "fig5b",
            formats=("png",),
            bbox_inches=None,
            pad_inches=0.0,
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
