"""Reusable Matplotlib panels, theme, composition, and deterministic export."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

from visualization.data import MeshData, ObservationChain, ScientificFigureError
from visualization.transforms import (
    SelectedTransferState,
    deterministic_spatial_subsample,
)


@dataclass(frozen=True)
class FigureTheme:
    """Central publication style with journal and blog presets."""

    name: str
    font_family: str
    base_font_size: float
    title_size: float
    axis_label_size: float
    panel_label_size: float
    line_width: float
    marker_size: float
    signed_colormap: str
    magnitude_colormap: str
    invalid_color: str
    mesh_color: str
    observation_colors: Mapping[str, str]
    raster_dpi: int
    pdf_font_type: int

    @classmethod
    def preset(cls, name: str) -> "FigureTheme":
        if name == "journal":
            return cls(
                name="journal",
                font_family="DejaVu Sans",
                base_font_size=8.5,
                title_size=11.0,
                axis_label_size=8.5,
                panel_label_size=11.0,
                line_width=1.4,
                marker_size=4.0,
                signed_colormap="RdBu_r",
                magnitude_colormap="viridis",
                invalid_color="#BDBDBD",
                mesh_color="#7F8C8D",
                observation_colors={"right": "#2166AC", "left": "#B2182B"},
                raster_dpi=300,
                pdf_font_type=42,
            )
        if name == "blog":
            return cls(
                name="blog",
                font_family="DejaVu Sans",
                base_font_size=10.0,
                title_size=14.0,
                axis_label_size=10.0,
                panel_label_size=13.0,
                line_width=2.0,
                marker_size=5.5,
                signed_colormap="RdBu_r",
                magnitude_colormap="viridis",
                invalid_color="#BDBDBD",
                mesh_color="#87939A",
                observation_colors={"right": "#287D91", "left": "#D95F02"},
                raster_dpi=240,
                pdf_font_type=42,
            )
        raise ScientificFigureError(f"unknown theme preset {name!r}")

    def apply(self) -> None:
        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.base_font_size,
                "figure.titlesize": self.title_size,
                "axes.labelsize": self.axis_label_size,
                "pdf.fonttype": self.pdf_font_type,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "savefig.bbox": "tight",
            }
        )


@dataclass(frozen=True)
class ScalePolicy:
    """Independent geometry, vector, and color scaling."""

    deformation_scale: float = 1.0
    arrow_scale: float = 1.0
    arrow_minimum_mm: float = 0.0
    color_limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.deformation_scale)
            or self.deformation_scale <= 0.0
            or not math.isfinite(self.arrow_scale)
            or self.arrow_scale <= 0.0
            or not math.isfinite(self.arrow_minimum_mm)
            or self.arrow_minimum_mm < 0.0
        ):
            raise ScientificFigureError("scale policy values are invalid")
        if self.color_limits is not None:
            low, high = self.color_limits
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ScientificFigureError("color limits are invalid")


@dataclass(frozen=True)
class SourceTable:
    """Deterministically serializable numerical source data."""

    filename: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    label_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.filename.endswith(".csv") or not self.columns:
            raise ScientificFigureError("source table filename/columns are invalid")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ScientificFigureError("source table row width is invalid")
        label_indices = {
            self.columns.index(name)
            for name in self.label_columns
            if name in self.columns
        }
        for row in self.rows:
            for index, value in enumerate(row):
                if index in label_indices:
                    continue
                if not math.isfinite(float(value)):
                    raise ScientificFigureError(
                        f"source table {self.filename} contains non-finite data"
                    )


@dataclass
class RenderedFigure:
    """In-memory figure plus source tables and provenance."""

    figure: plt.Figure
    basename: str
    figure_kind: str
    represented_variable: str
    units: str
    normalization: str
    design_ids: tuple[str, ...]
    mesh_ids: tuple[str, ...]
    cases: tuple[str, ...]
    xi_values: tuple[float, ...]
    indentation_values_mm: tuple[float, ...]
    coordinate_convention: Mapping[str, Any]
    interpolation: Mapping[str, Any]
    validity: Mapping[str, Any]
    scale_policy: ScalePolicy
    color_limits: tuple[float, float] | None
    panel_metadata: tuple[Mapping[str, Any], ...]
    source_tables: tuple[SourceTable, ...]
    notes: tuple[str, ...] = ()


class MeshPanel:
    """Render exact undeformed mesh topology without altering scientific data."""

    def render(
        self,
        axis: plt.Axes,
        mesh: MeshData,
        theme: FigureTheme,
        *,
        linewidth: float = 0.08,
        alpha: float = 0.18,
    ) -> dict[str, Any]:
        coordinates = mesh.coordinate_by_node_id
        polygons = [
            [coordinates[int(node_id)] for node_id in connectivity]
            for connectivity in mesh.element_connectivity
        ]
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors="none",
                edgecolors=theme.mesh_color,
                linewidths=linewidth,
                alpha=alpha,
                zorder=1,
            )
        )
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        return {
            "component": "MeshPanel",
            "mesh_id": mesh.mesh_id,
            "node_count": len(mesh.node_ids),
            "element_count": len(mesh.element_ids),
            "units": mesh.units,
            "aspect": "equal",
        }


class DeformedMeshPanel:
    """Render deformed topology only when a full nodal field is available."""

    def render(
        self,
        axis: plt.Axes,
        mesh: MeshData,
        displacement_by_node_id: Mapping[int, Sequence[float]],
        theme: FigureTheme,
        scale_policy: ScalePolicy,
    ) -> dict[str, Any]:
        if set(displacement_by_node_id) != set(mesh.node_ids):
            raise ScientificFigureError(
                "DeformedMeshPanel requires displacement for every mesh node"
            )
        coordinates = {
            node_id: mesh.coordinate_by_node_id[node_id]
            + scale_policy.deformation_scale
            * np.asarray(displacement_by_node_id[node_id], dtype=float)
            for node_id in mesh.node_ids
        }
        if not np.isfinite(np.asarray(list(coordinates.values()))).all():
            raise ScientificFigureError("deformed mesh coordinates are non-finite")
        polygons = [
            [coordinates[int(node_id)] for node_id in connectivity]
            for connectivity in mesh.element_connectivity
        ]
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors="#B7E1EA",
                edgecolors=theme.mesh_color,
                linewidths=0.10,
                alpha=0.55,
                zorder=2,
            )
        )
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        return {
            "component": "DeformedMeshPanel",
            "deformation_scale": scale_policy.deformation_scale,
            "full_nodal_field": True,
        }


class DisplacementVectorPanel:
    """Render deterministic sparse arrows representing actual u=[ux,uy]."""

    def render(
        self,
        axis: plt.Axes,
        coordinates: np.ndarray,
        displacement: np.ndarray,
        theme: FigureTheme,
        scale_policy: ScalePolicy,
        *,
        maximum_arrows: int,
        color: str = "#262626",
    ) -> tuple[dict[str, Any], np.ndarray]:
        points = np.asarray(coordinates, dtype=float)
        vectors = np.asarray(displacement, dtype=float)
        if points.shape != vectors.shape or points.ndim != 2 or points.shape[1] != 2:
            raise ScientificFigureError("vector panel requires matching [n,2] arrays")
        magnitudes = np.linalg.norm(vectors, axis=1)
        eligible = np.flatnonzero(magnitudes >= scale_policy.arrow_minimum_mm)
        if not len(eligible):
            raise ScientificFigureError("no displacement vector passes the threshold")
        local = deterministic_spatial_subsample(
            points[eligible], maximum_count=maximum_arrows
        )
        selected = eligible[local]
        axis.quiver(
            points[selected, 0],
            points[selected, 1],
            vectors[selected, 0],
            vectors[selected, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0 / scale_policy.arrow_scale,
            color=color,
            width=0.0045,
            headwidth=3.8,
            headlength=5.0,
            zorder=8,
        )
        return (
            {
                "component": "DisplacementVectorPanel",
                "represented_vector": "physical displacement u=[u_x,u_y]",
                "selection": "deterministic spatial binning",
                "selected_count": int(len(selected)),
                "candidate_count": int(len(points)),
                "arrow_scale": scale_policy.arrow_scale,
                "minimum_magnitude_mm": scale_policy.arrow_minimum_mm,
                "normalized_arrows": False,
            },
            selected,
        )


class ContactInputAnnotation:
    """Annotate the actual prescribed indentation direction and contact point."""

    def render(
        self,
        axis: plt.Axes,
        state: SelectedTransferState,
        *,
        length_mm: float = 3.0,
    ) -> dict[str, Any]:
        if state.contact_point_mm is None:
            return {
                "component": "ContactInputAnnotation",
                "available": False,
            }
        point = np.asarray(state.contact_point_mm)
        direction = np.asarray(state.indentation_direction)
        start = point - length_mm * direction
        axis.annotate(
            "",
            xy=point,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#111111",
                "lw": 2.0,
                "mutation_scale": 13,
            },
            zorder=12,
        )
        axis.scatter(*point, s=24, facecolor="white", edgecolor="black", zorder=13)
        return {
            "component": "ContactInputAnnotation",
            "available": True,
            "contact_point_mm": point.tolist(),
            "indentation_direction": direction.tolist(),
            "arrow_length_mm": length_mm,
            "represented_vector": "prescribed indentation direction, not displacement",
        }


class ObservationBoundaryOverlay:
    """Render independent reference/deformed chains without a center connection."""

    def render(
        self,
        axis: plt.Axes,
        chains: Mapping[str, ObservationChain],
        theme: FigureTheme,
        *,
        displacement_by_side: Mapping[str, np.ndarray] | None = None,
        values_by_side: Mapping[str, np.ndarray] | None = None,
        norm: colors.Normalize | None = None,
        scale_policy: ScalePolicy = ScalePolicy(),
    ) -> dict[str, Any]:
        for side in ("right", "left"):
            chain = chains[side]
            reference = chain.undeformed_coordinates
            axis.plot(
                reference[:, 0],
                reference[:, 1],
                color="#5F6368",
                lw=1.0,
                alpha=0.8,
                zorder=5,
            )
            if displacement_by_side is None:
                axis.plot(
                    reference[:, 0],
                    reference[:, 1],
                    color=theme.observation_colors[side],
                    lw=2.4,
                    label=f"{side} observation chain",
                    zorder=6,
                )
                continue
            displacement = np.asarray(displacement_by_side[side], dtype=float)
            if displacement.shape != reference.shape:
                raise ScientificFigureError("chain displacement shape is invalid")
            deformed = reference + scale_policy.deformation_scale * displacement
            segments = np.stack([deformed[:-1], deformed[1:]], axis=1)
            if values_by_side is not None and norm is not None:
                values = np.asarray(values_by_side[side], dtype=float)
                collection = LineCollection(
                    segments,
                    cmap=theme.signed_colormap,
                    norm=norm,
                    linewidths=3.0,
                    zorder=7,
                )
                collection.set_array(0.5 * (values[:-1] + values[1:]))
                axis.add_collection(collection)
            else:
                axis.plot(
                    deformed[:, 0],
                    deformed[:, 1],
                    color=theme.observation_colors[side],
                    lw=2.8,
                    zorder=7,
                )
        axis.set_aspect("equal", adjustable="box")
        return {
            "component": "ObservationBoundaryOverlay",
            "sides": ["right", "left"],
            "center_connected": False,
            "deformation_scale": scale_policy.deformation_scale,
            "contact_facing_surface_inferred": False,
        }


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


class FigureExporter:
    """Write PNG/PDF, deterministic source CSVs, and one strict manifest."""

    def export(
        self,
        rendered: RenderedFigure,
        output_directory: str | Path,
        *,
        formats: Sequence[str],
        dpi: int,
        serialized_spec: Mapping[str, Any],
        spec_path: str | None,
        source_artifacts: Sequence[str],
        source_checksums_sha256: Mapping[str, str],
        framework_version: str,
    ) -> dict[str, Any]:
        output = Path(output_directory).resolve()
        source_directory = output / "source_data"
        output.mkdir(parents=True, exist_ok=True)
        source_directory.mkdir(exist_ok=True)
        source_paths: list[str] = []
        for table in rendered.source_tables:
            path = source_directory / table.filename
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(table.columns)
                for row in table.rows:
                    writer.writerow(
                        f"{value:.17g}"
                        if isinstance(value, (float, np.floating))
                        else value
                        for value in row
                    )
            source_paths.append(str(path))
        outputs = []
        for extension in formats:
            if extension not in {"png", "pdf"}:
                raise ScientificFigureError(f"unsupported export format {extension}")
            path = output / f"{rendered.basename}.{extension}"
            if extension == "pdf":
                rendered.figure.savefig(
                    path,
                    metadata={
                        "Creator": "LIT Hand Scientific Figure Framework",
                        "Producer": "Matplotlib",
                        "CreationDate": None,
                        "ModDate": None,
                    },
                )
            else:
                rendered.figure.savefig(
                    path,
                    dpi=dpi,
                    metadata={"Software": "LIT Hand Scientific Figure Framework"},
                )
            outputs.append(
                {
                    "path": str(path),
                    "format": extension,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        plt.close(rendered.figure)
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
        manifest = {
            "figure_kind": rendered.figure_kind,
            "figure_spec_path": spec_path,
            "serialized_spec": serialized_spec,
            "source_artifacts": list(source_artifacts),
            "source_checksums_sha256": dict(source_checksums_sha256),
            "design_ids": list(rendered.design_ids),
            "mesh_ids": list(rendered.mesh_ids),
            "cases": list(rendered.cases),
            "xi_values": list(rendered.xi_values),
            "indentation_values_mm": list(rendered.indentation_values_mm),
            "represented_variable": rendered.represented_variable,
            "normalization": rendered.normalization,
            "units": rendered.units,
            "coordinate_convention": dict(rendered.coordinate_convention),
            "interpolation": dict(rendered.interpolation),
            "validity_masks": dict(rendered.validity),
            "deformation_scale": rendered.scale_policy.deformation_scale,
            "arrow_scale": rendered.scale_policy.arrow_scale,
            "arrow_minimum_mm": rendered.scale_policy.arrow_minimum_mm,
            "color_limits": list(rendered.color_limits)
            if rendered.color_limits is not None
            else None,
            "panel_metadata": list(rendered.panel_metadata),
            "source_data_files": source_paths,
            "output_paths": outputs,
            "framework_version": framework_version,
            "git_commit": git_commit,
            "notes": list(rendered.notes),
        }
        manifest_path = output / "plot_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {
            "figure_kind": rendered.figure_kind,
            "manifest": str(manifest_path),
            "source_data": source_paths,
            "outputs": outputs,
        }
