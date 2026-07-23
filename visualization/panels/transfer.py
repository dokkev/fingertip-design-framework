"""Transfer-map, metric, layout, colorbar, and labeling panels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from visualization.data import ScientificFigureError
from visualization.theme import FigureTheme
from visualization.transforms import SelectedTransferState

def _display_zeta(side: str, eta: np.ndarray, gap_width: float) -> np.ndarray:
    half = 0.5 * gap_width
    scale = 1.0 - half
    if side == "right":
        return (eta - 1.0) * scale - half
    if side == "left":
        return (1.0 - eta) * scale + half
    raise ScientificFigureError(f"unknown side {side!r}")


class TransferMapPanel:
    """Render discrete xi rows on two separately drawn sidewall segments."""

    def render(
        self,
        axis: plt.Axes,
        states: Sequence[SelectedTransferState],
        theme: FigureTheme,
        norm: colors.Normalize,
        *,
        gap_width: float = 0.08,
    ) -> dict[str, Any]:
        xis = np.asarray([state.xi for state in states])
        image = None
        for side in ("right", "left"):
            eta = np.asarray(states[0].eta_by_side[side])
            if any(
                not np.array_equal(eta, state.eta_by_side[side])
                for state in states[1:]
            ):
                raise ScientificFigureError("transfer-map eta grids disagree")
            values = np.asarray([state.values_by_side[side] for state in states])
            x = _display_zeta(side, eta, gap_width)
            image = axis.imshow(
                values,
                origin="lower",
                interpolation="none",
                aspect="auto",
                extent=(float(x.min()), float(x.max()), -0.5, len(xis) - 0.5),
                cmap=theme.signed_colormap,
                norm=norm,
            )
        half = 0.5 * gap_width
        axis.axvspan(
            -half,
            half,
            facecolor=theme.invalid_color,
            alpha=0.32,
            hatch="////",
            edgecolor="#666666",
            zorder=4,
        )
        for row in np.arange(0.5, len(xis) - 0.5, 1.0):
            axis.axhline(row, color="white", lw=0.5, alpha=0.9)
        axis.set_xlim(-1.02, 1.02)
        axis.set_yticks(range(len(xis)), [f"{xi:.2f}" for xi in xis])
        axis.set_xlabel(r"signed display coordinate $\zeta$ (center unsampled)")
        axis.set_ylabel(r"commanded contact location $\xi$")
        axis.text(
            0.01,
            1.01,
            "R bonded → crown-side",
            transform=axis.transAxes,
            ha="left",
            fontsize=6.5,
            color="0.3",
        )
        axis.text(
            0.99,
            1.01,
            "L crown-side → bonded",
            transform=axis.transAxes,
            ha="right",
            fontsize=6.5,
            color="0.3",
        )
        return {
            "component": "TransferMapPanel",
            "image": image,
            "xi_values": xis.tolist(),
            "xi_interpolation": False,
            "row_rendering": "discrete, interpolation=none",
            "center_gap_width_display": gap_width,
            "center_connected": False,
        }


class LocationDistanceMatrixPanel:
    """Render one symmetric location-distance matrix."""

    def render(
        self,
        axis: plt.Axes,
        matrix: np.ndarray,
        xi_values: Sequence[float],
        theme: FigureTheme,
        norm: colors.Normalize,
    ) -> dict[str, Any]:
        values = np.asarray(matrix, dtype=float)
        if values.shape != (len(xi_values), len(xi_values)):
            raise ScientificFigureError("distance matrix shape is invalid")
        image = axis.imshow(
            values, origin="lower", cmap=theme.magnitude_colormap, norm=norm
        )
        for row in range(len(xi_values)):
            for column in range(len(xi_values)):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black"
                    if norm(values[row, column]) < 0.72
                    else "white",
                )
        labels = [f"{value:.2f}" for value in xi_values]
        axis.set_xticks(range(len(labels)), labels, rotation=45)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel(r"$\xi_j$")
        axis.set_ylabel(r"$\xi_i$")
        return {
            "component": "LocationDistanceMatrixPanel",
            "image": image,
            "symmetric_max_error": float(np.max(np.abs(values - values.T))),
            "diagonal_max_abs": float(np.max(np.abs(np.diag(values)))),
        }


class ProfilePanel:
    """Render semantic profiles without joining the two sidewall chains."""

    def render(
        self,
        axis: plt.Axes,
        states: Sequence[SelectedTransferState],
        theme: FigureTheme,
    ) -> dict[str, Any]:
        xi_norm = colors.Normalize(
            vmin=min(state.xi for state in states),
            vmax=max(state.xi for state in states),
        )
        cmap = plt.get_cmap(theme.magnitude_colormap)
        for state in states:
            for side in ("right", "left"):
                axis.plot(
                    state.eta_by_side[side],
                    state.values_by_side[side],
                    color=cmap(xi_norm(state.xi)),
                    ls="-" if side == "right" else "--",
                    lw=theme.line_width,
                    label=rf"$\xi={state.xi:.2f}$"
                    if side == "right"
                    else None,
                )
        axis.axhline(0.0, color="0.55", lw=0.7)
        axis.set_xlabel(r"$\eta$ (solid: right, dashed: left)")
        axis.set_ylabel(f"{states[0].quantity} [{states[0].units}]")
        axis.legend(frameon=False, ncol=2)
        axis.grid(alpha=0.2)
        return {
            "component": "ProfilePanel",
            "center_connected": False,
            "side_encoding": "right solid; left dashed",
        }


class MetricSummaryPanel:
    """Render declarative scalar metrics as a compact table."""

    def render(
        self,
        axis: plt.Axes,
        metrics_by_design: Mapping[str, Mapping[str, float | None]],
    ) -> dict[str, Any]:
        axis.axis("off")
        lines = []
        for design_id, metrics in metrics_by_design.items():
            lines.append(f"{design_id}")
            for name, value in metrics.items():
                readable = name.replace("_", " ")
                lines.append(
                    f"  {readable}: "
                    + ("UNAVAILABLE" if value is None else f"{value:.5g}")
                )
        axis.text(
            0.02,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            family="monospace",
            bbox={"facecolor": "#F7F7F7", "edgecolor": "#BBBBBB", "pad": 8},
        )
        return {
            "component": "MetricSummaryPanel",
            "design_ids": list(metrics_by_design),
            "optical_observability_claim": False,
        }


class SharedColorbar:
    """Add one explicitly labeled colorbar for compared panels."""

    def add(
        self,
        figure: plt.Figure,
        image: Any,
        axes: Sequence[plt.Axes],
        *,
        label: str,
    ) -> dict[str, Any]:
        figure.colorbar(image, ax=list(axes), label=label, shrink=0.88)
        return {
            "component": "SharedColorbar",
            "label": label,
            "shared_across_panel_count": len(axes),
        }


class PanelLabelManager:
    """Place deterministic panel labels in axes coordinates."""

    def apply(
        self,
        axes: Sequence[plt.Axes],
        theme: FigureTheme,
        labels: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        actual = list(labels or [chr(ord("A") + index) for index in range(len(axes))])
        if len(actual) != len(axes):
            raise ScientificFigureError("panel-label count is invalid")
        for axis, label in zip(axes, actual):
            axis.text(
                -0.10,
                1.06,
                label,
                transform=axis.transAxes,
                fontsize=theme.panel_label_size,
                fontweight="bold",
                va="top",
                ha="left",
            )
        return {"component": "PanelLabelManager", "labels": actual}


class FigureComposer:
    """Own figure layout while panels own only axes rendering."""

    def create(
        self,
        rows: int,
        columns: int,
        *,
        figsize: tuple[float, float],
        title: str,
    ) -> tuple[plt.Figure, np.ndarray]:
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )
        figure.suptitle(title)
        return figure, axes

