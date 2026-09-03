"""Compose Figure 2: the LUMO morphology-optimization flow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
)

from lumo.fingertip import Fingertip  # noqa: E402
from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    add_figure_box,
    plot_fingertip_parameterization,
    publication_context,
    save_figure,
)


_ROOT = Path(__file__).resolve().parents[1]
_INPUT = (
    _ROOT
    / "output"
    / "validation"
    / "fingertip_production_objective_freeze"
    / "nominal_fingertip_objectives.npz"
)
_PANEL_DIRECTORY = _ROOT / "output" / "figures" / "figure2"
_OUTPUT_STEM = Path(__file__).resolve().with_suffix("")
_LOADED_SCENARIO = "sphere_15mm_y+0mm"
_LOADED_FORCE_N = 10.0
_OPTIX_X_LIMITS_MM = (-16.0, 16.0)
_OPTIX_Z_LIMITS_MM = (-19.0, 14.0)

_VARIABLE_FONT_RC = {
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Roman",
    "mathtext.it": "Nimbus Roman:italic",
    "mathtext.bf": "Nimbus Roman:bold",
    "mathtext.sf": "Nimbus Roman",
}

_PANEL_BOXES = (
    (0.012, 0.185, 0.254, 0.965),
    (0.282, 0.185, 0.512, 0.965),
    (0.540, 0.185, 0.750, 0.965),
    (0.778, 0.185, 0.988, 0.965),
)


def _load_frozen_state() -> dict[str, object]:
    """Load the frozen nominal checkpoint used by publication visualizations."""

    if not _INPUT.is_file():
        raise FileNotFoundError(f"missing frozen production state: {_INPUT}")
    with np.load(_INPUT, allow_pickle=False) as data:
        scenario_matches = np.flatnonzero(data["scenario_names"] == _LOADED_SCENARIO)
        force_matches = np.flatnonzero(
            np.isclose(data["force_targets_n"], _LOADED_FORCE_N)
        )
        if len(scenario_matches) != 1 or len(force_matches) != 1:
            raise RuntimeError("requested loaded production checkpoint is not unique")
        scenario_index = int(scenario_matches[0])
        force_index = int(force_matches[0])
        offset, count = np.asarray(
            data["contact_record_offsets"][scenario_index, force_index],
            dtype=np.int64,
        )
        return {
            "reference_vertices_m": np.asarray(
                data["reference_vertices_m"],
                dtype=np.float64,
            ),
            "tetrahedra": np.asarray(data["tet_indices"], dtype=np.int32),
            "loaded_vertices_m": np.asarray(
                data["silicone_vertices_m"][scenario_index, force_index],
                dtype=np.float64,
            ),
            "contact_positions_m": np.asarray(
                data["contact_positions_W_m"][offset : offset + count],
                dtype=np.float64,
            ),
            "actual_force_n": float(
                data["actual_forces_n"][scenario_index, force_index]
            ),
            "indentation_mm": 1.0e3
            * float(data["indentations_m"][scenario_index, force_index]),
            "sphere_diameter_mm": float(
                data["sphere_diameters_mm"][scenario_index]
            ),
        }


def _load_force_displacement() -> tuple[np.ndarray, np.ndarray]:
    """Return the actual frozen force-displacement checkpoints for panel (b)."""

    with np.load(_INPUT, allow_pickle=False) as data:
        matches = np.flatnonzero(data["scenario_names"] == _LOADED_SCENARIO)
        if len(matches) != 1:
            raise RuntimeError("force-displacement scenario is not unique")
        scenario_index = int(matches[0])
        indentation_mm = 1.0e3 * np.asarray(
            data["indentations_m"][scenario_index],
            dtype=np.float64,
        )
        force_n = np.asarray(
            data["actual_forces_n"][scenario_index],
            dtype=np.float64,
        )
    return (
        np.concatenate(([0.0], indentation_mm)),
        np.concatenate(([0.0], force_n)),
    )


def _panel_image(filename: str) -> np.ndarray:
    path = _PANEL_DIRECTORY / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing Figure 2 panel asset: {path}")
    return plt.imread(path)


def _crop_white_margin(
    image: np.ndarray,
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    crop_reference = image if reference is None else reference
    visible = np.any(crop_reference[..., :3] < 0.985, axis=2)
    rows, columns = np.nonzero(visible)
    if rows.size == 0:
        return image
    padding = 2
    row_start = max(0, int(rows.min()) - padding)
    row_end = min(image.shape[0], int(rows.max()) + padding + 1)
    column_start = max(0, int(columns.min()) - padding)
    column_end = min(image.shape[1], int(columns.max()) + padding + 1)
    return image[row_start:row_end, column_start:column_end]


def _color_optix_led_source(
    image: np.ndarray,
    fingertip: Fingertip,
) -> np.ndarray:
    """Color the finite-area LED package consistently with the legend."""

    colored = image.copy()
    height, width = colored.shape[:2]
    led = fingertip.parameters.led
    source_z_mm = 1.0e3 * fingertip.led_source_centers_m[2][2]
    x_min, x_max = _OPTIX_X_LIMITS_MM
    z_min, z_max = _OPTIX_Z_LIMITS_MM
    left = int(round((-0.5 * led.width_mm - x_min) / (x_max - x_min) * width))
    right = int(round((0.5 * led.width_mm - x_min) / (x_max - x_min) * width))
    top = int(round((z_max - source_z_mm - led.height_mm) / (z_max - z_min) * height))
    bottom = int(round((z_max - source_z_mm) / (z_max - z_min) * height))
    colored[top:bottom, left:right, :3] = matplotlib.colors.to_rgb(
        DEFAULT_STYLE.colors.optical
    )
    if colored.shape[2] == 4:
        colored[top:bottom, left:right, 3] = 1.0
    return colored


def _emphasize_side_escape_rays(image: np.ndarray) -> np.ndarray:
    """Strengthen only existing optical-ray pixels at the lateral boundaries."""

    emphasized = image.copy()
    rgb = emphasized[..., :3]
    _, width = rgb.shape[:2]
    columns = np.arange(width)[None, :]
    lateral_region = (columns < 0.25 * width) | (columns > 0.75 * width)
    green_ray = (
        (rgb[..., 1] - rgb[..., 0] > 0.10)
        & (rgb[..., 1] - rgb[..., 2] > 0.04)
        & (rgb[..., 1] < 0.96)
    )
    selected = green_ray & lateral_region
    optical_color = np.asarray(
        matplotlib.colors.to_rgb(DEFAULT_STYLE.colors.optical),
        dtype=np.float64,
    )
    rgb[selected] = 0.25 * rgb[selected] + 0.75 * optical_color
    return emphasized


def _add_panel_frame(
    figure: plt.Figure,
    box: tuple[float, float, float, float],
    label: str,
    title: str,
    subtitle: str,
) -> None:
    x0, y0, x1, y1 = box
    add_figure_box(figure, box)
    figure.text(
        x0 + 0.012,
        y1 - 0.024,
        f"({label})",
        ha="left",
        va="top",
        fontsize=7.7,
        fontweight="bold",
        color="#111111",
    )
    figure.text(
        x0 + 0.038,
        y1 - 0.024,
        title,
        ha="left",
        va="top",
        fontsize=7.7,
        color="#111111",
    )
    figure.text(
        0.5 * (x0 + x1),
        y1 - 0.063,
        subtitle,
        ha="center",
        va="top",
        fontsize=7.0,
        fontstyle="italic",
        color="#252525",
    )


def _add_flow_arrow(
    figure: plt.Figure,
    left_box: tuple[float, float, float, float],
    right_box: tuple[float, float, float, float],
) -> None:
    y = 0.56
    gap_left = left_box[2]
    gap_right = right_box[0]
    figure.add_artist(
        FancyArrowPatch(
            (gap_left + 0.003, y),
            (gap_right - 0.003, y),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=6.5,
            linewidth=0.8,
            color="#666666",
            clip_on=False,
            zorder=50,
        )
    )


def _draw_parameterization(figure: plt.Figure, fingertip: Fingertip) -> None:
    x0, y0, x1, y1 = _PANEL_BOXES[0]
    axis = figure.add_axes((x0 + 0.004, y0 + 0.135, x1 - x0 - 0.008, y1 - y0 - 0.205))
    parameter_style = replace(
        DEFAULT_STYLE,
        axis_label_font_size_pt=7.2,
        tick_font_size_pt=7.0,
        line_width_pt=0.95,
        spine_width_pt=0.55,
        tick_width_pt=0.55,
        tick_length_pt=2.0,
    )
    plot_fingertip_parameterization(
        axis,
        fingertip,
        show_legend=False,
        show_fixed_dimensions=False,
        style=parameter_style,
    )
    axis.set_xlim(-20.5, 20.5)
    axis.set_ylim(-20.4, 13.7)
    axis.grid(False)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_axis_off()

    figure.add_artist(
        FancyBboxPatch(
            (x0 + 0.020, y0 + 0.025),
            x1 - x0 - 0.040,
            0.085,
            boxstyle="round,pad=0.005,rounding_size=0.006",
            transform=figure.transFigure,
            facecolor="#FFFFFF",
            edgecolor="#222222",
            linewidth=0.55,
        )
    )
    figure.text(
        0.5 * (x0 + x1),
        y0 + 0.083,
        "Design vector",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#222222",
    )
    figure.text(
        0.5 * (x0 + x1),
        y0 + 0.048,
        r"$\boldsymbol{\theta}=[h_{\rm fp},h_{\rm ep},w_s,h_s,w_v]$",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#222222",
    )


def _draw_mechanics(figure: plt.Figure) -> None:
    x0, y0, x1, y1 = _PANEL_BOXES[1]

    render_axis = figure.add_axes((x0 + 0.004, y0 + 0.305, x1 - x0 - 0.008, 0.370))
    render_axis.imshow(_panel_image("c_newton_mechanics.png"))
    render_axis.set_axis_off()
    force_x = x0 + 0.55 * (x1 - x0)
    figure.add_artist(
        FancyArrowPatch(
            (force_x, y0 + 0.420),
            (force_x, y0 + 0.500),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=1.1,
            color=DEFAULT_STYLE.colors.optimization,
            zorder=25,
        )
    )
    figure.text(
        force_x - 0.008,
        y0 + 0.460,
        r"$F_{\mathrm{ext}}$",
        ha="right",
        va="center",
        fontsize=7.0,
        color=DEFAULT_STYLE.colors.optimization,
    )

    checkpoint_displacement_mm, checkpoint_force_n = _load_force_displacement()
    displacement_limit_mm = float(checkpoint_displacement_mm[-1])
    force_limit_n = float(checkpoint_force_n[-1])
    displacement_mm = np.linspace(0.0, displacement_limit_mm, 160)
    exponential_shape = 1.6
    force_n = force_limit_n * np.expm1(
        exponential_shape * displacement_mm / displacement_limit_mm
    ) / np.expm1(exponential_shape)
    curve_axis = figure.add_axes((x0 + 0.018, y0 + 0.120, x1 - x0 - 0.036, 0.150))
    curve_axis.plot(
        displacement_mm,
        force_n,
        color=DEFAULT_STYLE.colors.mechanical,
        linewidth=1.35,
    )
    curve_axis.fill_between(
        displacement_mm,
        force_n,
        color=DEFAULT_STYLE.colors.mechanical,
        alpha=0.08,
    )
    curve_axis.set_xlabel("Displacement [mm]", fontsize=7.0, labelpad=1.0)
    curve_axis.set_ylabel("Force [N]", fontsize=7.0, labelpad=1.0)
    curve_axis.set_xticks(())
    curve_axis.set_yticks(())
    curve_axis.set_xlim(left=0.0)
    curve_axis.set_ylim(bottom=0.0)
    curve_axis.text(
        0.05,
        0.88,
        "progressive stiffening",
        transform=curve_axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        fontstyle="italic",
        color=DEFAULT_STYLE.colors.mechanical,
    )
    for spine in curve_axis.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.55)
    figure.text(
        0.5 * (x0 + x1),
        y0 + 0.040,
        r"Mechanical Response $\rightarrow J_{\mathrm{contact}}$",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#222222",
    )


def _draw_optics(figure: plt.Figure, fingertip: Fingertip) -> None:
    x0, y0, x1, y1 = _PANEL_BOXES[2]
    image_height = 0.268
    image_width = x1 - x0 - 0.016
    unloaded_axis = figure.add_axes((x0 + 0.008, y0 + 0.405, image_width, image_height))
    loaded_axis = figure.add_axes((x0 + 0.008, y0 + 0.115, image_width, image_height))
    unloaded_image = _color_optix_led_source(
        _panel_image("d_unloaded_optix.png"),
        fingertip,
    )
    loaded_image = _color_optix_led_source(
        _panel_image("d_loaded_optix.png"),
        fingertip,
    )
    for axis, image, label in (
        (unloaded_axis, _crop_white_margin(unloaded_image), "Unloaded"),
        (
            loaded_axis,
            _crop_white_margin(loaded_image, reference=unloaded_image),
            "Loaded",
        ),
    ):
        axis.imshow(_emphasize_side_escape_rays(image))
        axis.set_axis_off()
        axis.text(
            0.03,
            0.96,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.06",
                "facecolor": "white",
                "edgecolor": "none",
                "linewidth": 0.0,
                "alpha": 0.92,
            },
        )
    figure.text(
        0.5 * (x0 + x1),
        y0 + 0.040,
        r"Optical Response $\rightarrow J_{\mathrm{obs}}$",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#222222",
    )


def _nondominated(points: np.ndarray) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        dominates = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        keep[index] = not np.any(dominates)
    return keep


def _draw_bayesian_optimization(figure: plt.Figure) -> None:
    x0, y0, x1, y1 = _PANEL_BOXES[3]

    rng = np.random.default_rng(20260830)
    evaluated = np.column_stack(
        (
            rng.uniform(0.25, 0.88, 18),
            rng.uniform(0.18, 0.86, 18),
        )
    )
    evaluated[:5] = np.asarray(
        ((0.30, 0.79), (0.45, 0.71), (0.58, 0.66), (0.72, 0.54), (0.84, 0.39))
    )
    proposed = np.asarray((0.74, 0.73))
    bo_axis = figure.add_axes((x0 + 0.014, y0 + 0.300, x1 - x0 - 0.028, 0.390))
    xx, yy = np.meshgrid(np.linspace(0.15, 0.95, 100), np.linspace(0.10, 0.95, 100))
    field = np.exp(-((xx - 0.73) ** 2 / 0.055 + (yy - 0.72) ** 2 / 0.075))
    field += 0.45 * np.exp(-((xx - 0.40) ** 2 / 0.08 + (yy - 0.40) ** 2 / 0.09))
    bo_axis.contourf(xx, yy, field, levels=10, cmap="Blues", alpha=0.25)
    bo_axis.contour(
        xx,
        yy,
        field,
        levels=6,
        colors="#8FA8BC",
        linewidths=0.25,
        alpha=0.45,
    )
    bo_axis.scatter(
        evaluated[:, 0],
        evaluated[:, 1],
        s=10,
        c="#555A60",
        edgecolors="white",
        linewidths=0.25,
        label="Evaluated designs",
        zorder=4,
    )
    pareto_points = evaluated[_nondominated(evaluated)]
    pareto_order = np.argsort(pareto_points[:, 0])
    bo_axis.plot(
        pareto_points[pareto_order, 0],
        pareto_points[pareto_order, 1],
        color="#1F5A91",
        linewidth=0.7,
        label="Pareto front",
        zorder=5,
    )
    bo_axis.scatter(
        proposed[0],
        proposed[1],
        marker="*",
        s=55,
        c="#F57C00",
        edgecolors="white",
        linewidths=0.4,
        label="Next query",
        zorder=6,
    )
    bo_axis.set_xlabel(r"$J_{\mathrm{contact}}$", fontsize=7.0, labelpad=1.0)
    bo_axis.set_ylabel(r"$J_{\mathrm{obs}}$", fontsize=7.0, labelpad=1.0)
    bo_axis.set_xticks(())
    bo_axis.set_yticks(())
    bo_axis.set_xlim(0.15, 0.95)
    bo_axis.set_ylim(0.10, 0.95)
    bo_axis.legend(
        loc="lower left",
        fontsize=5.4,
        frameon=True,
        framealpha=0.92,
        borderpad=0.18,
        handlelength=1.35,
        handletextpad=0.28,
        labelspacing=0.16,
        markerscale=0.80,
    )
    for spine in bo_axis.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.5)

    figure.add_artist(
        FancyArrowPatch(
            (0.5 * (x0 + x1), y0 + 0.255),
            (0.5 * (x0 + x1), y0 + 0.123),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.8,
            color="#333333",
        )
    )
    figure.add_artist(
        FancyBboxPatch(
            (x0 + 0.030, y0 + 0.025),
            x1 - x0 - 0.060,
            0.075,
            boxstyle="round,pad=0.004,rounding_size=0.005",
            transform=figure.transFigure,
            facecolor="#FFFFFF",
            edgecolor="#222222",
            linewidth=0.65,
        )
    )
    figure.text(
        0.5 * (x0 + x1),
        y0 + 0.0625,
        "Propose next morphology\n" + r"$\boldsymbol{\theta}_{i+1}$",
        ha="center",
        va="center",
        fontsize=5.8,
        color="#222222",
        linespacing=0.90,
    )


def _add_feedback_loop(figure: plt.Figure) -> None:
    left_box = _PANEL_BOXES[0]
    right_box = _PANEL_BOXES[-1]
    y = 0.125
    figure.add_artist(
        Line2D(
            (right_box[0] + 0.5 * (right_box[2] - right_box[0]), right_box[0] + 0.5 * (right_box[2] - right_box[0]), left_box[0] + 0.5 * (left_box[2] - left_box[0])),
            (right_box[1], y, y),
            transform=figure.transFigure,
            color="#333333",
            linewidth=0.9,
            linestyle=(0, (4.0, 3.0)),
            clip_on=False,
            zorder=30,
        )
    )
    figure.add_artist(
        FancyArrowPatch(
            (left_box[0] + 0.5 * (left_box[2] - left_box[0]), y),
            (left_box[0] + 0.5 * (left_box[2] - left_box[0]), left_box[1] - 0.001),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=0.9,
            linestyle=(0, (4.0, 3.0)),
            color="#333333",
            clip_on=False,
            zorder=30,
        )
    )
    figure.text(
        0.5,
        y - 0.014,
        "Update morphology and iterate",
        ha="center",
        va="top",
        fontsize=6.0,
        fontstyle="italic",
        color="#333333",
    )


def _add_shared_legend(figure: plt.Figure) -> None:
    handles = (
        Patch(facecolor=DEFAULT_STYLE.colors.carrier, edgecolor="#34383C", label="Rigid carrier"),
        Patch(facecolor=DEFAULT_STYLE.colors.silicone, edgecolor="#777777", label="Deformable pad"),
        Patch(
            facecolor=DEFAULT_STYLE.colors.optical,
            edgecolor="#087A49",
            label="LED / light source",
        ),
        Line2D((), (), color=DEFAULT_STYLE.colors.mechanical, linestyle="--", linewidth=1.0, label="Bonding surface"),
        Line2D((), (), marker="o", linestyle="none", markerfacecolor="#8B8B8B", markeredgecolor="#555555", markersize=5.5, label="Spherical indenter"),
        Line2D((), (), color="#008C67", linewidth=1.0, label="Optical ray path"),
    )
    legend = figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.002),
        ncol=6,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#A8A8A8",
        fontsize=5.9,
        handlelength=1.35,
        handleheight=0.75,
        handletextpad=0.40,
        columnspacing=0.75,
        borderpad=0.25,
        labelspacing=0.20,
    )
    legend.get_frame().set_linewidth(0.35)


def main() -> None:
    """Assemble the four-stage publication flow without rerunning simulation."""

    _load_frozen_state()
    nominal_fingertip = Fingertip()
    schematic_geometry = replace(
        nominal_fingertip.parameters.geometry,
        flat_pad_height_mm=7.0,
        semiellipse_height_mm=12.0,
        stem_height_mm=8.5,
        void_width_mm=3.5,
    )
    fingertip = Fingertip(
        parameters=replace(
            nominal_fingertip.parameters,
            geometry=schematic_geometry,
        )
    )
    with publication_context(), matplotlib.rc_context(rc=_VARIABLE_FONT_RC):
        figure = plt.figure(figsize=(7.16, 3.75))
        for box, label, title, subtitle in zip(
            _PANEL_BOXES,
            ("a", "b", "c", "d"),
            (
                "Parameterized Morphology",
                "Newton Soft Mechanics",
                "OptiX Light Transport",
                "Bayesian Optimization",
            ),
            (
                "",
                "",
                "",
                "",
            ),
            strict=True,
        ):
            _add_panel_frame(figure, box, label, title, subtitle)

        _draw_parameterization(figure, fingertip)
        _draw_mechanics(figure)
        _draw_optics(figure, nominal_fingertip)
        _draw_bayesian_optimization(figure)
        for left_box, right_box in zip(_PANEL_BOXES[:-1], _PANEL_BOXES[1:], strict=True):
            _add_flow_arrow(figure, left_box, right_box)
        _add_feedback_loop(figure)
        _add_shared_legend(figure)
        outputs = save_figure(
            figure,
            _OUTPUT_STEM,
            formats=("pdf", "svg", "png"),
        )
        plt.close(figure)

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
