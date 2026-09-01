"""Compose Figure 3 from the saved hybrid-mechanics ablation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from lumo.fingertip import LED_RECESS_DEPTH_MM, LED_RECESS_WIDTH_MM  # noqa: E402
from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    publication_context,
    save_figure,
)


_ROOT = Path(__file__).resolve().parents[2]
_INPUT_DIRECTORY = _ROOT / "output" / "validation" / "hybrid_mechanics_ablation"
_INPUT_PATH = _INPUT_DIRECTORY / "ablation_results.npz"
_OUTPUT_STEM = _ROOT / "output" / "figures" / "figure_3_hybrid_mechanics_ablation"

_CASE_NAMES = ("soft_only", "bonded_t", "lumo")
_DISPLAY_NAMES = {
    "soft_only": "Soft-only",
    "bonded_t": "Bonded-T",
    "lumo": "LUMO",
}
_CASE_STYLE = {
    "soft_only": ("#7A7A7A", (0, (1.0, 1.5))),
    "bonded_t": ("#7651A8", (0, (5.0, 2.0))),
    "lumo": ("#174A73", "-"),
}
_GAP_STYLE = {
    "near_zero": ("#777777", (0, (1.0, 1.5)), "0.01 mm"),
    "nominal": (DEFAULT_STYLE.colors.optical, "-", "0.19 mm (nominal)"),
    "large": ("#CC6677", (0, (5.0, 2.0)), "0.50 mm"),
}


def _outer_boundary(
    *,
    half_width_mm: float,
    ellipse_center_z_mm: float,
    ellipse_height_mm: float,
    top_z_mm: float,
) -> np.ndarray:
    angles = np.linspace(0.0, np.pi, 257)
    ellipse = np.column_stack(
        (
            half_width_mm * np.cos(angles),
            ellipse_center_z_mm - ellipse_height_mm * np.sin(angles),
        )
    )
    return np.vstack(
        (
            (-half_width_mm, top_z_mm),
            (-half_width_mm, ellipse_center_z_mm),
            ellipse[::-1][1:-1],
            (half_width_mm, ellipse_center_z_mm),
            (half_width_mm, top_z_mm),
        )
    )


def _draw_morphology(
    axis: plt.Axes,
    case_name: str,
    morphology_mm: np.ndarray,
) -> None:
    flat_height, ellipse_height, stem_width, stem_height, void_width = morphology_mm
    half_width = 15.0
    top_z = 10.0
    ellipse_center = -flat_height
    pad = Polygon(
        _outer_boundary(
            half_width_mm=half_width,
            ellipse_center_z_mm=ellipse_center,
            ellipse_height_mm=ellipse_height,
            top_z_mm=top_z,
        ),
        closed=True,
        facecolor=DEFAULT_STYLE.colors.silicone,
        edgecolor="#777777",
        linewidth=0.65,
        zorder=1,
    )
    axis.add_patch(pad)

    if case_name != "soft_only":
        cavity_half_width = 0.5 * stem_width + (
            0.0 if case_name == "bonded_t" else void_width
        )
        axis.add_patch(
            Rectangle(
                (-cavity_half_width, -stem_height),
                2.0 * cavity_half_width,
                stem_height,
                facecolor="white",
                edgecolor="none",
                zorder=2,
            )
        )
        carrier_boundary = np.asarray(
            (
                (-half_width, 8.0),
                (-10.0, 8.0),
                (-10.0, 0.0),
                (-0.5 * stem_width, 0.0),
                (-0.5 * stem_width, -stem_height),
                (0.5 * stem_width, -stem_height),
                (0.5 * stem_width, 0.0),
                (10.0, 0.0),
                (10.0, 8.0),
                (half_width, 8.0),
                (half_width, top_z),
                (-half_width, top_z),
            )
        )
        axis.add_patch(
            Polygon(
                carrier_boundary,
                closed=True,
                facecolor=DEFAULT_STYLE.colors.carrier,
                edgecolor="#34383C",
                linewidth=0.7,
                zorder=3,
            )
        )
        recess_depth = LED_RECESS_DEPTH_MM
        axis.add_patch(
            Rectangle(
                (-0.5 * LED_RECESS_WIDTH_MM, -stem_height),
                LED_RECESS_WIDTH_MM,
                recess_depth,
                facecolor="white",
                edgecolor="none",
                zorder=4,
            )
        )
        axis.plot(
            (-0.9, 0.9),
            (-stem_height + recess_depth,) * 2,
            color=DEFAULT_STYLE.colors.optical,
            linewidth=2.0,
            solid_capstyle="butt",
            zorder=5,
        )
        if case_name == "bonded_t":
            axis.plot(
                (-0.5 * stem_width, -0.5 * stem_width),
                (0.0, -stem_height),
                color=DEFAULT_STYLE.colors.mechanical,
                linewidth=1.0,
            )
            axis.plot(
                (0.5 * stem_width, 0.5 * stem_width),
                (-stem_height, 0.0),
                color=DEFAULT_STYLE.colors.mechanical,
                linewidth=1.0,
            )
            axis.plot(
                (-0.5 * stem_width, 0.5 * stem_width),
                (-stem_height, -stem_height),
                color=DEFAULT_STYLE.colors.mechanical,
                linewidth=1.0,
            )

    axis.set_xlim(-16.0, 16.0)
    axis.set_ylim(ellipse_center - ellipse_height - 0.8, top_z + 0.8)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def _crop_white(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3]
    visible = np.any(rgb < 0.985, axis=2)
    rows, columns = np.nonzero(visible)
    if not len(rows):
        return image
    padding = 3
    return image[
        max(0, rows.min() - padding) : min(image.shape[0], rows.max() + padding + 1),
        max(0, columns.min() - padding) : min(image.shape[1], columns.max() + padding + 1),
    ]


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=2.0, width=0.55, pad=1.5)
    axis.grid(False)


def _plot_cases(axis: plt.Axes, data: np.lib.npyio.NpzFile, x_key: str, y_key: str) -> None:
    for case_name in _CASE_NAMES:
        color, linestyle = _CASE_STYLE[case_name]
        axis.plot(
            data[f"{case_name}_{x_key}"],
            data[f"{case_name}_{y_key}"],
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
            label=_DISPLAY_NAMES[case_name],
        )


def main() -> None:
    if not _INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"missing ablation results; run simulation_ablation_study.py: {_INPUT_PATH}"
        )
    with np.load(_INPUT_PATH, allow_pickle=False) as data, publication_context():
        required = tuple(
            f"gap_{label}_state_distance" for label in _GAP_STYLE
        )
        missing = tuple(name for name in required if name not in data.files)
        if missing:
            raise RuntimeError(
                "gap sensitivity is missing; rerun simulation_ablation_study.py "
                f"--optics ({', '.join(missing)})"
            )
        figure = plt.figure(figsize=(7.16, 6.05))
        outer = figure.add_gridspec(
            3,
            2,
            height_ratios=(1.62, 1.0, 1.0),
            hspace=0.42,
            wspace=0.27,
            left=0.072,
            right=0.985,
            bottom=0.075,
            top=0.975,
        )

        structure_axis = figure.add_subplot(outer[0, 0])
        structure_axis.set_axis_off()
        structure_axis.set_title(
            "(a)  Structural comparison",
            loc="left",
            fontsize=8.0,
            fontweight="bold",
            pad=2.0,
        )
        morphology = data["morphology_mm"]
        for index, case_name in enumerate(_CASE_NAMES):
            axis = structure_axis.inset_axes((index / 3.0, 0.10, 0.32, 0.82))
            _draw_morphology(axis, case_name, morphology)
            axis.set_title(_DISPLAY_NAMES[case_name], fontsize=6.7, pad=0.5)
            axis.text(
                0.5,
                -0.01,
                rf"$J_{{\rm contact}}={float(data[f'{case_name}_q_contact']):.3f}$",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=5.6,
            )
            if case_name == "lumo":
                axis.text(
                    0.5,
                    -0.09,
                    rf"$w_{{\rm void}}={morphology[4]:g}$ mm, "
                    rf"$g_{{\rm eff}}={LED_RECESS_DEPTH_MM:g}$ mm",
                    transform=axis.transAxes,
                    color=DEFAULT_STYLE.colors.optical,
                    ha="center",
                    va="top",
                    fontsize=4.8,
                )

        states_axis = figure.add_subplot(outer[0, 1])
        states_axis.set_axis_off()
        states_axis.set_title(
            "(b)  Matched Newton states",
            loc="left",
            fontsize=8.0,
            fontweight="bold",
            pad=2.0,
        )
        for row, force_n in enumerate((2, 10)):
            for column, case_name in enumerate(_CASE_NAMES):
                axis = states_axis.inset_axes(
                    (0.02 + 0.325 * column, 0.50 - 0.43 * row, 0.31, 0.40)
                )
                image = plt.imread(
                    _INPUT_DIRECTORY / f"newton_{case_name}_{force_n}n.png"
                )
                axis.imshow(_crop_white(image))
                axis.set_axis_off()
                if row == 0:
                    axis.set_title(_DISPLAY_NAMES[case_name], fontsize=5.8, pad=0.2)
                if column == 0:
                    axis.text(
                        -0.03,
                        0.5,
                        f"{force_n} N",
                        transform=axis.transAxes,
                        rotation=90,
                        ha="right",
                        va="center",
                        fontsize=5.6,
                        fontweight="bold",
                    )
        stress_map = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.PowerNorm(
                gamma=0.30,
                vmin=0.0,
                vmax=100.0,
                clip=True,
            ),
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                "mechanical_stress",
                (
                    (0.00, (0.95, 0.94, 0.88)),
                    (0.20, (1.00, 0.91, 0.55)),
                    (0.38, (0.99, 0.72, 0.27)),
                    (0.55, (0.96, 0.40, 0.12)),
                    (0.72, (0.82, 0.10, 0.08)),
                    (1.00, (0.48, 0.00, 0.06)),
                ),
            ),
        )
        colorbar_axis = states_axis.inset_axes((0.25, 0.005, 0.50, 0.025))
        colorbar = figure.colorbar(
            stress_map,
            cax=colorbar_axis,
            orientation="horizontal",
            ticks=(0, 25, 50, 100),
        )
        colorbar.ax.tick_params(labelsize=4.8, length=1.2, pad=0.5)
        colorbar.set_label("Elastic von Mises stress [kPa]", fontsize=5.2, labelpad=0.5)

        contact_axis = figure.add_subplot(outer[1, 0])
        force_axis = figure.add_subplot(outer[1, 1])
        optical_axis = figure.add_subplot(outer[2, 0])
        gap_axis = figure.add_subplot(outer[2, 1])

        for case_name in _CASE_NAMES:
            color, linestyle = _CASE_STYLE[case_name]
            force = data[f"{case_name}_force_n"]
            contact_axis.plot(
                force,
                1.0e6 * data[f"{case_name}_external_area_m2"],
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
            )
            force_axis.plot(
                1.0e3 * data[f"{case_name}_indentation_m"],
                force,
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
            )
            optical_axis.plot(
                data[f"{case_name}_optical_force_n"],
                data[f"{case_name}_optical_state_distance"],
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
            )
        contact_axis.axvspan(0.0, 2.0, color="#F4C27A", alpha=0.15, zorder=-5)

        stiffness_inset = force_axis.inset_axes((0.54, 0.10, 0.42, 0.47))
        for case_name in _CASE_NAMES:
            color, linestyle = _CASE_STYLE[case_name]
            stiffness_inset.plot(
                data[f"{case_name}_force_n"],
                1.0e-3 * data[f"{case_name}_incremental_stiffness_n_m"],
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
            )
        stiffness_inset.set_xlabel("Force [N]", fontsize=4.7, labelpad=0.5)
        stiffness_inset.set_ylabel(r"$K_{\rm inc}$ [N/mm]", fontsize=4.7, labelpad=0.5)
        stiffness_inset.tick_params(labelsize=4.3, length=1.2, pad=0.8)
        stiffness_inset.spines[["top", "right"]].set_visible(False)

        visible_inset = optical_axis.inset_axes((0.52, 0.12, 0.44, 0.43))
        for case_name in _CASE_NAMES:
            color, linestyle = _CASE_STYLE[case_name]
            visible_inset.plot(
                data[f"{case_name}_optical_force_n"],
                100.0 * data[f"{case_name}_optical_relative_visible_power_change"],
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
            )
        visible_inset.axhline(0.0, color="#AAAAAA", linewidth=0.45)
        visible_inset.set_xlabel("Force [N]", fontsize=4.7, labelpad=0.5)
        visible_inset.set_ylabel(r"$\Delta P_{\rm vis}/P_0$ [%]", fontsize=4.7, labelpad=0.5)
        visible_inset.tick_params(labelsize=4.3, length=1.2, pad=0.8)
        visible_inset.spines[["top", "right"]].set_visible(False)

        for label, (color, linestyle, display) in _GAP_STYLE.items():
            gap_axis.plot(
                data[f"gap_{label}_force_n"],
                data[f"gap_{label}_state_distance"],
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
                marker="o",
                markersize=2.3,
                label=display,
            )
        gap_axis.axvspan(0.0, 2.0, color="#F4C27A", alpha=0.15, zorder=-5)
        low_load_inset = gap_axis.inset_axes((0.53, 0.12, 0.43, 0.45))
        for label in ("nominal", "large"):
            color, linestyle, _ = _GAP_STYLE[label]
            force = data[f"gap_{label}_force_n"]
            distance = data[f"gap_{label}_state_distance"]
            low_load_inset.plot(
                force[:3],
                distance[:3],
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                marker="o",
                markersize=1.8,
            )
        low_load_inset.set_xlim(-0.05, 2.25)
        low_load_inset.set_ylim(0.0, 0.0021)
        low_load_inset.set_xlabel("0.19/0.50 mm, low load [N]", fontsize=4.5, labelpad=0.5)
        low_load_inset.set_ylabel(r"$D(F)$", fontsize=4.7, labelpad=0.5)
        low_load_inset.tick_params(labelsize=4.3, length=1.2, pad=0.8)
        low_load_inset.spines[["top", "right"]].set_visible(False)

        contact_axis.set_title("(c)  External contact progression", loc="left", fontsize=7.2, pad=2.0)
        contact_axis.set_xlabel("Force [N]", labelpad=1.0)
        contact_axis.set_ylabel(r"$A_{\rm ext}$ [mm$^2$]", labelpad=1.0)
        force_axis.set_title("(d)  Mechanical response", loc="left", fontsize=7.2, pad=2.0)
        force_axis.set_xlabel(r"Indentation $\delta$ [mm]", labelpad=1.0)
        force_axis.set_ylabel("Force [N]", labelpad=1.0)
        optical_axis.set_title("(e)  Controlled optical response", loc="left", fontsize=7.2, pad=2.0)
        optical_axis.set_xlabel("Force [N]", labelpad=1.0)
        optical_axis.set_ylabel(r"$D(F)=\|(\mathbf{y}_F-\mathbf{y}_0)/5\|_2$", labelpad=1.0)
        gap_axis.set_title("(f)  Effective-gap sensitivity (fixed LUMO states)", loc="left", fontsize=7.2, pad=2.0)
        gap_axis.set_xlabel("Force [N]", labelpad=1.0)
        gap_axis.set_ylabel(r"$D(F)$", labelpad=1.0)
        for axis in (contact_axis, force_axis, optical_axis, gap_axis):
            _style_axis(axis)
            axis.tick_params(labelsize=5.8)
        contact_axis.set_xlim(0.0, 10.6)
        optical_axis.set_xlim(0.0, 10.6)
        optical_axis.set_ylim(bottom=0.0)
        gap_axis.set_xlim(0.0, 10.6)
        gap_axis.set_ylim(bottom=0.0)
        gap_axis.legend(
            loc="upper left",
            frameon=False,
            fontsize=5.5,
            handlelength=1.8,
            borderaxespad=0.2,
        )

        legend_handles = tuple(
            Line2D(
                (),
                (),
                color=_CASE_STYLE[name][0],
                linestyle=_CASE_STYLE[name][1],
                linewidth=1.3,
                label=_DISPLAY_NAMES[name],
            )
            for name in _CASE_NAMES
        )
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol=3,
            frameon=False,
            fontsize=6.3,
            handlelength=2.0,
            columnspacing=1.4,
        )
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
