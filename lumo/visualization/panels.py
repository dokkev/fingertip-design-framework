"""Small axes-owned scientific panels for LUMO publication figures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from .style import (
    DEFAULT_STYLE,
    STATUS_MARKERS,
    PublicationStyle,
    material_color,
)


def _finite_1d(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _paired_arrays(
    x_name: str,
    x_values: Sequence[float] | np.ndarray,
    y_name: str,
    y_values: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = _finite_1d(x_name, x_values)
    y_array = _finite_1d(y_name, y_values)
    if x_array.shape != y_array.shape:
        raise ValueError(f"{x_name} and {y_name} must have the same shape")
    return x_array, y_array


def _style_axes(
    axes: Axes,
    *,
    grid: bool,
    style: PublicationStyle,
) -> None:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for spine in (axes.spines["left"], axes.spines["bottom"]):
        spine.set_linewidth(style.spine_width_pt)
    axes.tick_params(
        width=style.tick_width_pt,
        length=style.tick_length_pt,
        labelsize=style.tick_font_size_pt,
        direction="out",
    )
    axes.grid(
        grid,
        color=style.colors.grid,
        linewidth=style.grid_width_pt,
        alpha=0.55,
    )
    axes.set_axisbelow(True)


def _plot_single_curve(
    axes: Axes,
    x_values: Sequence[float] | np.ndarray,
    y_values: Sequence[float] | np.ndarray,
    *,
    x_name: str,
    y_name: str,
    x_label: str,
    y_label: str,
    color: str,
    label: str | None,
    marker: str,
    line_style: str,
    grid: bool,
    style: PublicationStyle,
) -> tuple[Artist, ...]:
    x_array, y_array = _paired_arrays(x_name, x_values, y_name, y_values)
    (line,) = axes.plot(
        x_array,
        y_array,
        color=color,
        linestyle=line_style,
        marker=marker,
        linewidth=style.line_width_pt,
        markersize=style.marker_size_pt,
        label=label,
    )
    axes.set_xlabel(x_label, fontsize=style.axis_label_font_size_pt)
    axes.set_ylabel(y_label, fontsize=style.axis_label_font_size_pt)
    _style_axes(axes, grid=grid, style=style)
    if label is not None:
        axes.legend(frameon=False, fontsize=style.legend_font_size_pt)
    return (line,)


def plot_pareto(
    axes: Axes,
    j_contact: Sequence[float] | np.ndarray,
    j_obs: Sequence[float] | np.ndarray,
    *,
    material: str,
    status: Sequence[str] | str | None = None,
    pareto_mask: Sequence[bool] | np.ndarray | None = None,
    label: str | None = None,
    grid: bool = True,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Artist, ...]:
    """Plot one material's objective samples using status-specific markers."""

    contact, observation = _paired_arrays("j_contact", j_contact, "j_obs", j_obs)
    count = contact.size
    if status is None:
        statuses = np.full(count, "candidate", dtype=object)
    elif isinstance(status, str):
        statuses = np.full(count, status, dtype=object)
    else:
        statuses = np.asarray(status, dtype=object)
        if statuses.shape != contact.shape:
            raise ValueError("status must have one entry per objective sample")
    unknown = sorted(set(statuses) - set(STATUS_MARKERS))
    if unknown:
        raise ValueError(f"unsupported design status: {unknown}")

    if pareto_mask is None:
        pareto = np.zeros(count, dtype=bool)
    else:
        pareto = np.asarray(pareto_mask, dtype=bool)
        if pareto.shape != contact.shape:
            raise ValueError("pareto_mask must have one entry per objective sample")

    color = material_color(material, style)
    material_label = label or material.replace("_", " ").title()
    artists: list[Artist] = []
    if np.any(pareto):
        order = np.argsort(contact[pareto])
        (frontier,) = axes.plot(
            contact[pareto][order],
            observation[pareto][order],
            color=color,
            linewidth=style.line_width_pt,
            alpha=0.8,
            zorder=2,
        )
        artists.append(frontier)

    unique_statuses = tuple(dict.fromkeys(str(value) for value in statuses))
    for status_name in unique_statuses:
        mask = statuses == status_name
        legend_label = (
            material_label
            if len(unique_statuses) == 1
            else f"{material_label}, {status_name}"
        )
        collection = axes.scatter(
            contact[mask],
            observation[mask],
            s=(1.5 * style.marker_size_pt) ** 2,
            marker=STATUS_MARKERS[status_name],
            color=color,
            edgecolor=style.colors.carrier,
            linewidth=0.45,
            alpha=0.88,
            label=legend_label,
            zorder=3,
        )
        artists.append(collection)

    axes.set_xlabel(r"$J_{\mathrm{contact}}$", fontsize=style.axis_label_font_size_pt)
    axes.set_ylabel(r"$J_{\mathrm{obs}}$", fontsize=style.axis_label_font_size_pt)
    _style_axes(axes, grid=grid, style=style)
    axes.legend(frameon=False, fontsize=style.legend_font_size_pt)
    return tuple(artists)


def plot_force_displacement(
    axes: Axes,
    indentation_mm: Sequence[float] | np.ndarray,
    force_n: Sequence[float] | np.ndarray,
    *,
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
    line_style: str = "-",
    grid: bool = True,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Artist, ...]:
    """Plot external force against indentation depth."""

    return _plot_single_curve(
        axes,
        indentation_mm,
        force_n,
        x_name="indentation_mm",
        y_name="force_n",
        x_label=r"Indentation $\delta$ [mm]",
        y_label=r"$F_{\mathrm{ext}}$ [N]",
        color=color or style.colors.mechanical,
        label=label,
        marker=marker,
        line_style=line_style,
        grid=grid,
        style=style,
    )


def plot_contact_area(
    axes: Axes,
    force_n: Sequence[float] | np.ndarray,
    contact_area_mm2: Sequence[float] | np.ndarray,
    *,
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
    line_style: str = "-",
    grid: bool = True,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Artist, ...]:
    """Plot contact-patch area against external load."""

    return _plot_single_curve(
        axes,
        force_n,
        contact_area_mm2,
        x_name="force_n",
        y_name="contact_area_mm2",
        x_label=r"$F_{\mathrm{ext}}$ [N]",
        y_label=r"Contact area [mm$^2$]",
        color=color or style.colors.mechanical,
        label=label,
        marker=marker,
        line_style=line_style,
        grid=grid,
        style=style,
    )


def plot_incremental_stiffness(
    axes: Axes,
    force_n: Sequence[float] | np.ndarray,
    stiffness_n_mm: Sequence[float] | np.ndarray,
    *,
    label: str | None = None,
    color: str | None = None,
    marker: str = "s",
    line_style: str = "-",
    grid: bool = True,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Artist, ...]:
    """Plot incremental stiffness against external load."""

    return _plot_single_curve(
        axes,
        force_n,
        stiffness_n_mm,
        x_name="force_n",
        y_name="stiffness_n_mm",
        x_label=r"$F_{\mathrm{ext}}$ [N]",
        y_label=r"$K_{\mathrm{inc}}$ [N/mm]",
        color=color or style.colors.mechanical,
        label=label,
        marker=marker,
        line_style=line_style,
        grid=grid,
        style=style,
    )


def plot_optical_response(
    axes: Axes,
    force_n: Sequence[float] | np.ndarray,
    response: Sequence[float] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    grid: bool = True,
    style: PublicationStyle = DEFAULT_STYLE,
) -> tuple[Artist, ...]:
    """Plot one or more optical-response channels against load."""

    force = _finite_1d("force_n", force_n)
    response_array = np.asarray(response, dtype=np.float64)
    if response_array.ndim == 1:
        response_array = response_array[:, None]
    if response_array.ndim != 2 or response_array.shape[0] != force.size:
        raise ValueError("response must have shape (force, channel)")
    if not np.all(np.isfinite(response_array)):
        raise ValueError("response must contain only finite values")
    channel_count = response_array.shape[1]
    if labels is not None and len(labels) != channel_count:
        raise ValueError("labels must have one entry per response channel")

    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v")
    artists: list[Artist] = []
    for channel_index in range(channel_count):
        (line,) = axes.plot(
            force,
            response_array[:, channel_index],
            color=style.colors.optical,
            linestyle=line_styles[channel_index % len(line_styles)],
            marker=markers[channel_index % len(markers)],
            linewidth=style.line_width_pt,
            markersize=style.marker_size_pt,
            alpha=max(0.45, 1.0 - 0.1 * channel_index),
            label=None if labels is None else labels[channel_index],
        )
        artists.append(line)

    axes.set_xlabel(r"$F_{\mathrm{ext}}$ [N]", fontsize=style.axis_label_font_size_pt)
    axes.set_ylabel("Normalized optical response", fontsize=style.axis_label_font_size_pt)
    _style_axes(axes, grid=grid, style=style)
    if labels is not None:
        axes.legend(frameon=False, fontsize=style.legend_font_size_pt)
    return tuple(artists)


def _crop_white_border(image: np.ndarray, threshold: float) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("whitespace_threshold must be between zero and one")
    values = np.asarray(image)
    if values.ndim not in {2, 3}:
        raise ValueError("image must be grayscale, RGB, or RGBA")
    normalized = values.astype(np.float64)
    if np.issubdtype(values.dtype, np.integer):
        normalized /= np.iinfo(values.dtype).max
    elif normalized.size and normalized.max() > 1.0:
        normalized /= 255.0

    if normalized.ndim == 2:
        content = normalized < threshold
    else:
        rgb = normalized[..., :3]
        content = np.any(rgb < threshold, axis=-1)
        if normalized.shape[-1] == 4:
            content &= normalized[..., 3] > 0.0
    occupied = np.argwhere(content)
    if occupied.size == 0:
        return values
    lower = occupied.min(axis=0)
    upper = occupied.max(axis=0) + 1
    return values[lower[0] : upper[0], lower[1] : upper[1]]


def plot_image(
    axes: Axes,
    image: np.ndarray,
    *,
    crop_whitespace: bool = False,
    whitespace_threshold: float = 0.995,
    style: PublicationStyle = DEFAULT_STYLE,
) -> Artist:
    """Insert a prepared image into an axes while preserving pixel aspect ratio."""

    del style
    image_array = np.asarray(image)
    if crop_whitespace:
        image_array = _crop_white_border(image_array, whitespace_threshold)
    artist = axes.imshow(image_array, aspect="equal", interpolation="none")
    axes.set_axis_off()
    return artist
