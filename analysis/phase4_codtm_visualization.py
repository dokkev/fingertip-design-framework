"""Create publication-ready static figures from immutable Phase 4K artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fem.codtm_visualization import (
    CODTMDataset,
    CODTMVisualizationError,
    common_eta_profiles,
    descriptor_verified_mask,
    display_zeta_for_side,
    finite_csv_audit,
    independent_tangent_gain,
    input_checksums,
    load_codtm_dataset,
    location_distance_matrix,
    mirror_metrics,
    profile_comparison_metrics,
    shape_distance_matrix,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "output" / "phase4_mechanical_transfer_map"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "output" / "phase4_codtm_visualization"
DEFAULT_INDENTATIONS = (0.25, 0.50, 1.00, 1.50)
XI_RANGE = (0.20, 0.80)
DISPLAY_GAP = 0.08


def _strict_write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                f"{value:.17g}" if isinstance(value, (float, np.floating)) else value
                for value in row
            )


def _slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _semantic_eta(dataset: CODTMDataset) -> np.ndarray:
    return np.stack(
        [dataset.eta_for_side(side) for side in dataset.side_order], axis=0
    )


def _case_by_xi(dataset: CODTMDataset, mesh: str) -> dict[float, str]:
    mapping: dict[float, str] = {}
    for case in dataset.cases_for_mesh(mesh):
        key = round(case.xi_cmd, 12)
        if key in mapping:
            raise CODTMVisualizationError(f"duplicate {mesh} xi={case.xi_cmd}")
        mapping[key] = case.name
    return mapping


def _selected_field(
    dataset: CODTMDataset,
    mesh: str,
    target: float,
    field: str,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, tuple[str, ...]]:
    cases = dataset.cases_for_mesh(mesh)
    values = []
    selections = []
    for case in cases:
        selection = dataset.select_case_field(case.name, field, target)
        values.append(selection.values)
        selections.append(
            {
                "case": case.name,
                "mesh": mesh,
                "xi_cmd": case.xi_cmd,
                **selection.metadata(),
            }
        )
    return (
        np.asarray(values, dtype=float),
        selections,
        np.asarray([case.xi_cmd for case in cases], dtype=float),
        tuple(case.name for case in cases),
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 12,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def _xi_color(xi: float):
    return plt.get_cmap("viridis")(
        colors.Normalize(vmin=XI_RANGE[0], vmax=XI_RANGE[1])(xi)
    )


def _save_figure(
    figure: plt.Figure,
    stem: str,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[dict[str, Any]]:
    outputs = []
    for extension in formats:
        path = figures_dir / f"{stem}.{extension}"
        if extension == "pdf":
            figure.savefig(
                path,
                metadata={
                    "Creator": "LIT Hand Phase 4K-Viz",
                    "Producer": "Matplotlib",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
        else:
            figure.savefig(
                path,
                dpi=dpi,
                metadata={"Software": "LIT Hand Phase 4K-Viz"},
            )
        outputs.append(
            {
                "path": str(path),
                "format": extension,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    plt.close(figure)
    return outputs


def _heatmap_panel(
    axis: plt.Axes,
    values: np.ndarray,
    xis: np.ndarray,
    eta: np.ndarray,
    side_order: Sequence[str],
    norm: colors.Normalize,
    *,
    cmap: str,
    title: str,
) -> list[Any]:
    artists = []
    for semantic_side in ("right", "left"):
        side_index = tuple(side_order).index(semantic_side)
        x = display_zeta_for_side(
            semantic_side, eta[side_index], gap_width=DISPLAY_GAP
        )
        image = axis.imshow(
            values[:, side_index, :],
            origin="lower",
            interpolation="none",
            aspect="auto",
            extent=(float(x.min()), float(x.max()), -0.5, len(xis) - 0.5),
            cmap=cmap,
            norm=norm,
        )
        artists.append(image)
    half = 0.5 * DISPLAY_GAP
    axis.axvspan(-half, half, facecolor="0.92", hatch="////", edgecolor="0.65")
    axis.axvline(-half, color="0.35", lw=0.7)
    axis.axvline(half, color="0.35", lw=0.7)
    for row in np.arange(0.5, len(xis) - 0.5, 1.0):
        axis.axhline(row, color="white", lw=0.5, alpha=0.8)
    axis.set_yticks(range(len(xis)), [f"{xi:.2f}" for xi in xis])
    axis.set_xlim(-1.02, 1.02)
    axis.set_title(title)
    axis.set_xlabel(r"signed display coordinate $\zeta$ (gap is unsampled)")
    axis.set_ylabel(r"commanded contact location $\xi$")
    axis.text(
        0.01,
        1.01,
        "R bonded → R crown-side",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="0.3",
    )
    axis.text(
        0.99,
        1.01,
        "L crown-side → L bonded",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="0.3",
    )
    return artists


def _spatial_atlas(
    dataset: CODTMDataset,
    mesh: str,
    indentations: Sequence[float],
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[dict[str, Any]], list[list[Any]], list[dict[str, Any]]]:
    eta = _semantic_eta(dataset)
    selected = []
    metadata = []
    xis = None
    for target in indentations:
        values, selections, location, _ = _selected_field(
            dataset, mesh, target, "u_normal"
        )
        selected.append(values)
        metadata.extend(selections)
        if xis is None:
            xis = location
        elif not np.array_equal(xis, location):
            raise CODTMVisualizationError("location order changed between indentations")
    all_values = np.asarray(selected)
    vlim = float(np.max(np.abs(all_values)))
    norm = colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    image = None
    for axis, target, values in zip(axes.ravel(), indentations, selected):
        image = _heatmap_panel(
            axis,
            values,
            np.asarray(xis),
            eta,
            dataset.side_order,
            norm,
            cmap="RdBu_r",
            title=rf"$\delta={target:.2f}$ mm",
        )[0]
    figure.suptitle(f"CODTM spatial atlas — {mesh} mesh, raw outward displacement")
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), label=r"$u_\mathrm{normal}$ [mm]")
    figure.text(
        0.5,
        -0.01,
        "Each row is a computed location; right and left are independent material "
        "chains. The hatched center is not observed and is not interpolated.",
        ha="center",
    )
    outputs = _save_figure(
        figure, "codtm_spatial_atlas", figures_dir, formats, dpi
    )
    rows = []
    for target, values in zip(indentations, selected):
        for location_index, xi in enumerate(xis):
            for side in ("right", "left"):
                side_index = dataset.side_index(side)
                for sample, (eta_value, displacement) in enumerate(
                    zip(eta[side_index], values[location_index, side_index])
                ):
                    rows.append(
                        [
                            mesh,
                            target,
                            xi,
                            side,
                            sample,
                            eta_value,
                            display_zeta_for_side(
                                side, [eta_value], gap_width=DISPLAY_GAP
                            )[0],
                            displacement,
                        ]
                    )
    return outputs, rows, [
        {
            "name": "codtm_spatial_atlas",
            "field": "u_normal",
            "field_kind": "raw",
            "units": "mm",
            "color_limits": [-vlim, vlim],
            "location_interpolation": "forbidden; discrete rows",
            "center_gap": "unsampled; two separately rendered side chains",
            "input_selections": metadata,
        }
    ]


def _profiles_and_physical(
    dataset: CODTMDataset,
    mesh: str,
    target: float,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[dict[str, Any]], list[list[Any]], list[list[Any]], list[dict[str, Any]]]:
    values, selections, xis, case_names = _selected_field(
        dataset, mesh, target, "u_normal"
    )
    u_xy, _, _, _ = _selected_field(dataset, mesh, target, "u_xy")
    eta = _semantic_eta(dataset)
    profile_rows = []
    physical_rows = []
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), sharey=True)
    for side_axis, side in zip(axes, ("right", "left")):
        side_index = dataset.side_index(side)
        for location_index, xi in enumerate(xis):
            side_axis.plot(
                eta[side_index],
                values[location_index, side_index],
                color=_xi_color(float(xi)),
                lw=1.7,
                label=rf"$\xi={xi:.2f}$",
            )
        side_axis.axhline(0.0, color="0.6", lw=0.7)
        side_axis.set_title(f"{side.capitalize()} observation sidewall")
        side_axis.set_xlabel(r"material coordinate $\eta$ (bonded 0 → crownward 1)")
        side_axis.grid(alpha=0.22)
    axes[0].set_ylabel(r"outward displacement $u_\mathrm{normal}$ [mm]")
    axes[1].legend(frameon=False, ncol=1)
    figure.suptitle(rf"CODTM profiles at $\delta={target:.2f}$ mm — {mesh} mesh")
    figure.text(
        0.5,
        -0.01,
        "The two panels are independent chains; no crown point or center connection is implied.",
        ha="center",
    )
    outputs = _save_figure(
        figure, "codtm_profiles_delta_1p50mm", figures_dir, formats, dpi
    )
    physical, axes_physical = plt.subplots(
        1, len(xis), figsize=(3.0 * len(xis), 2.8), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes_physical)
    for location_index, (axis, xi, case) in enumerate(
        zip(axes_array, xis, case_names)
    ):
        for side in ("right", "left"):
            side_index = dataset.side_index(side)
            reference = dataset.reference_xy[(case, side)]
            displacement = u_xy[location_index, side_index]
            deformed = reference + displacement
            axis.plot(reference[:, 0], reference[:, 1], color="0.65", lw=1.0)
            axis.plot(
                deformed[:, 0],
                deformed[:, 1],
                color=_xi_color(float(xi)),
                lw=2.0,
            )
            stride = max(1, len(reference) // 8)
            axis.quiver(
                reference[::stride, 0],
                reference[::stride, 1],
                displacement[::stride, 0],
                displacement[::stride, 1],
                color=_xi_color(float(xi)),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.006,
                alpha=0.65,
            )
            for sample, eta_value in enumerate(eta[side_index]):
                profile_rows.append(
                    [
                        mesh,
                        target,
                        xi,
                        side,
                        sample,
                        eta_value,
                        values[location_index, side_index, sample],
                        u_xy[location_index, side_index, sample, 0],
                        u_xy[location_index, side_index, sample, 1],
                    ]
                )
                physical_rows.append(
                    [
                        mesh,
                        target,
                        xi,
                        side,
                        sample,
                        eta_value,
                        reference[sample, 0],
                        reference[sample, 1],
                        displacement[sample, 0],
                        displacement[sample, 1],
                        deformed[sample, 0],
                        deformed[sample, 1],
                    ]
                )
        axis.annotate(
            "",
            xy=(0.50, 0.77),
            xytext=(0.50, 0.93),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
        )
        axis.text(0.53, 0.86, "indent", transform=axis.transAxes, fontsize=7)
        axis.set_title(rf"$\xi={xi:.2f}$")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [mm]")
        if location_index == 0:
            axis.set_ylabel("y [mm]")
    physical.suptitle(
        rf"Observation-sidewall deformation at $\delta={target:.2f}$ mm "
        "(1× physical scale; gray: reference)"
    )
    outputs += _save_figure(
        physical,
        "sidewall_deformation_delta_1p50mm",
        figures_dir,
        formats,
        dpi,
    )
    manifest = [
        {
            "name": "codtm_profiles_delta_1p50mm",
            "field": "u_normal",
            "field_kind": "raw",
            "units": "mm",
            "input_selections": selections,
        },
        {
            "name": "sidewall_deformation_delta_1p50mm",
            "field": "u_xy",
            "field_kind": "raw physical displacement",
            "units": "mm",
            "deformation_scale": 1.0,
            "aspect": "equal",
            "scope": "observation sidewalls only; no inferred contact-facing surface",
            "input_selections": selections,
        },
    ]
    return outputs, profile_rows, physical_rows, manifest


def _secant_atlas(
    dataset: CODTMDataset,
    mesh: str,
    indentations: Sequence[float],
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[dict[str, Any]], list[list[Any]], list[dict[str, Any]]]:
    eta = _semantic_eta(dataset)
    panels = []
    selections_all = []
    xis = None
    for target in indentations:
        values, selections, current_xis, _ = _selected_field(
            dataset, mesh, target, "G_secant"
        )
        panels.append(values)
        selections_all.extend(selections)
        xis = current_xis if xis is None else xis
    all_values = np.asarray(panels)
    vlim = float(np.max(np.abs(all_values)))
    norm = colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    image = None
    for axis, target, values in zip(axes.ravel(), indentations, panels):
        image = _heatmap_panel(
            axis,
            values,
            np.asarray(xis),
            eta,
            dataset.side_order,
            norm,
            cmap="PuOr_r",
            title=rf"$\delta={target:.2f}$ mm",
        )[0]
    figure.suptitle("CODTM secant transfer gain — normalization shown explicitly")
    assert image is not None
    figure.colorbar(
        image, ax=axes.ravel().tolist(), label=r"$G_\mathrm{secant}=u_n/\delta$ [–]"
    )
    outputs = _save_figure(
        figure, "codtm_secant_gain_atlas", figures_dir, formats, dpi
    )
    rows = []
    for target, values in zip(indentations, panels):
        for location_index, xi in enumerate(xis):
            for side in ("right", "left"):
                side_index = dataset.side_index(side)
                for sample, (eta_value, gain) in enumerate(
                    zip(eta[side_index], values[location_index, side_index])
                ):
                    rows.append(
                        [mesh, target, xi, side, sample, eta_value, gain]
                    )
    return outputs, rows, [
        {
            "name": "codtm_secant_gain_atlas",
            "field": "G_secant",
            "field_kind": "stored normalized field",
            "definition": "u_normal / delta_n",
            "units": "dimensionless",
            "color_limits": [-vlim, vlim],
            "input_selections": selections_all,
        }
    ]


def _distance_figures(
    dataset: CODTMDataset,
    mesh: str,
    indentations: Sequence[float],
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[
    list[dict[str, Any]],
    list[list[Any]],
    list[list[Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    eta = _semantic_eta(dataset)
    matrices = []
    selections = []
    xis = None
    fields = []
    for target in indentations:
        field, selection, current_xis, _ = _selected_field(
            dataset, mesh, target, "u_normal"
        )
        matrices.append(location_distance_matrix(field, eta))
        fields.append(field)
        selections.extend(selection)
        xis = current_xis if xis is None else xis
    vmax = float(np.max(matrices))
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 7.5), constrained_layout=True)
    image = None
    for axis, target, matrix in zip(axes.ravel(), indentations, matrices):
        image = axis.imshow(matrix, origin="lower", vmin=0.0, vmax=vmax, cmap="magma")
        for i in range(len(xis)):
            for j in range(len(xis)):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6.5)
        axis.set_xticks(range(len(xis)), [f"{xi:.2f}" for xi in xis], rotation=45)
        axis.set_yticks(range(len(xis)), [f"{xi:.2f}" for xi in xis])
        axis.set_title(rf"$\delta={target:.2f}$ mm")
        axis.set_xlabel(r"$\xi_j$")
        axis.set_ylabel(r"$\xi_i$")
    figure.suptitle("Pairwise CODTM distance — side integrals summed, gap excluded")
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), label=r"$D_{ij}$ [mm]")
    outputs = _save_figure(
        figure, "location_distance_matrices", figures_dir, formats, dpi
    )
    distance_rows = []
    for target, matrix in zip(indentations, matrices):
        for first, xi_i in enumerate(xis):
            for second, xi_j in enumerate(xis):
                distance_rows.append(
                    [mesh, target, xi_i, xi_j, matrix[first, second]]
                )
    shape = shape_distance_matrix(fields[-1], eta)
    shape_figure, axis = plt.subplots(figsize=(5.3, 4.6), constrained_layout=True)
    shape_image = axis.imshow(shape, origin="lower", vmin=0.0, cmap="cividis")
    for i in range(len(xis)):
        for j in range(len(xis)):
            axis.text(j, i, f"{shape[i, j]:.3f}", ha="center", va="center", fontsize=7)
    axis.set_xticks(range(len(xis)), [f"{xi:.2f}" for xi in xis], rotation=45)
    axis.set_yticks(range(len(xis)), [f"{xi:.2f}" for xi in xis])
    axis.set_xlabel(r"$\xi_j$")
    axis.set_ylabel(r"$\xi_i$")
    axis.set_title(r"Amplitude-normalized shape distance at $\delta=1.50$ mm")
    shape_figure.colorbar(shape_image, ax=axis, label=r"$D^\mathrm{shape}_{ij}$ [–]")
    outputs += _save_figure(
        shape_figure, "shape_distance_delta_1p50mm", figures_dir, formats, dpi
    )
    shape_rows = [
        [mesh, indentations[-1], xi_i, xi_j, shape[i, j]]
        for i, xi_i in enumerate(xis)
        for j, xi_j in enumerate(xis)
    ]
    offdiag = ~np.eye(len(xis), dtype=bool)
    metrics = {
        "raw_distance_1p50mm_offdiagonal_min_mm": float(matrices[-1][offdiag].min()),
        "raw_distance_1p50mm_offdiagonal_max_mm": float(matrices[-1][offdiag].max()),
        "shape_distance_1p50mm_offdiagonal_min": float(shape[offdiag].min()),
        "shape_distance_1p50mm_offdiagonal_max": float(shape[offdiag].max()),
        "distance_symmetry_max_error": float(
            max(np.max(np.abs(matrix - matrix.T)) for matrix in matrices)
        ),
        "distance_diagonal_max_abs": float(
            max(np.max(np.abs(np.diag(matrix))) for matrix in matrices)
        ),
    }
    manifests = [
        {
            "name": "location_distance_matrices",
            "field": "u_normal",
            "field_kind": "derived raw-signature metric",
            "units": "mm",
            "integration": "trapezoidal eta integral on each side separately, then summed",
            "common_color_limits": [0.0, vmax],
            "input_selections": selections,
        },
        {
            "name": "shape_distance_delta_1p50mm",
            "field": "u_normal",
            "field_kind": "amplitude-normalized signature metric",
            "units": "dimensionless",
            "zero_norm_guard": 1.0e-12,
            "input_selections": selections[-len(xis) :],
        },
    ]
    return outputs, distance_rows, shape_rows, metrics, manifests


def _mirror_figure(
    dataset: CODTMDataset,
    mesh: str,
    target: float,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[dict[str, Any]], list[list[Any]], dict[str, Any], list[dict[str, Any]]]:
    field, selections, xis, _ = _selected_field(dataset, mesh, target, "u_normal")
    eta = _semantic_eta(dataset)
    by_xi = {round(float(xi), 12): index for index, xi in enumerate(xis)}
    pairs = []
    for first, second in ((0.20, 0.80), (0.35, 0.65), (0.50, 0.50)):
        if round(first, 12) in by_xi and round(second, 12) in by_xi:
            pairs.append((first, second))
    figure, axes = plt.subplots(len(pairs), 2, figsize=(10.0, 2.8 * len(pairs)), squeeze=False)
    rows = []
    metric_rows = []
    for row_index, (first_xi, second_xi) in enumerate(pairs):
        first = field[by_xi[round(first_xi, 12)]]
        second = field[by_xi[round(second_xi, 12)]]
        result = mirror_metrics(first, second, eta, dataset.side_order)
        metric_rows.append(
            {
                "xi": first_xi,
                "mirror_xi": second_xi,
                "absolute_l2_mm": result["absolute_l2_mm"],
                "relative_l2": result["relative_l2"],
                "max_abs_mm": result["max_abs_mm"],
            }
        )
        for side in ("right", "left"):
            side_index = dataset.side_index(side)
            axes[row_index, 0].plot(
                eta[side_index],
                first[side_index],
                color="C0" if side == "right" else "C1",
                label=f"original {side}",
            )
            axes[row_index, 0].plot(
                eta[side_index],
                result["mirrored"][side_index],
                color="C0" if side == "right" else "C1",
                ls="--",
                label=f"mirrored {side}",
            )
            axes[row_index, 1].plot(
                eta[side_index],
                result["residual"][side_index],
                color="C0" if side == "right" else "C1",
                label=side,
            )
            for sample, eta_value in enumerate(eta[side_index]):
                rows.append(
                    [
                        mesh,
                        target,
                        first_xi,
                        second_xi,
                        side,
                        sample,
                        eta_value,
                        first[side_index, sample],
                        result["mirrored"][side_index, sample],
                        result["residual"][side_index, sample],
                        result["absolute_l2_mm"],
                        result["relative_l2"],
                        result["max_abs_mm"],
                    ]
                )
        axes[row_index, 0].set_title(
            rf"$\xi={first_xi:.2f}$ vs mirrored $\xi={second_xi:.2f}$"
        )
        axes[row_index, 1].set_title(
            f"residual; rel L2={100*result['relative_l2']:.2f}%, "
            f"max={result['max_abs_mm']:.4f} mm"
        )
        for axis in axes[row_index]:
            axis.axhline(0.0, color="0.6", lw=0.7)
            axis.set_xlabel(r"$\eta$")
            axis.grid(alpha=0.2)
        axes[row_index, 0].set_ylabel(r"$u_n$ [mm]")
        axes[row_index, 1].set_ylabel("residual [mm]")
    if pairs:
        axes[0, 0].legend(frameon=False, ncol=2)
        axes[0, 1].legend(frameon=False)
    figure.suptitle("Mirror-symmetry diagnostic — side swap, eta preserved")
    figure.tight_layout()
    outputs = _save_figure(
        figure, "mirror_symmetry_delta_1p50mm", figures_dir, formats, dpi
    )
    maximum_relative = max(row["relative_l2"] for row in metric_rows)
    judgement = (
        "CONSISTENT"
        if maximum_relative <= 0.05
        else "ASYMMETRIC"
        if math.isfinite(maximum_relative)
        else "UNRESOLVED"
    )
    metrics = {
        "status": judgement,
        "pairs": metric_rows,
        "classification_note": (
            "5% is a visualization diagnostic convention, not an optical "
            "observability or solver acceptance threshold."
        ),
    }
    return outputs, rows, metrics, [
        {
            "name": "mirror_symmetry_delta_1p50mm",
            "field": "u_normal",
            "field_kind": "valid displacement diagnostic",
            "mapping": "semantic side swap with eta preserved",
            "contact_pressure_closure_used": False,
            "input_selections": selections,
        }
    ]


def _tangent_figure(
    dataset: CODTMDataset,
    mesh: str,
    indentations: Sequence[float],
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[dict[str, Any]], list[list[Any]], dict[str, Any], list[dict[str, Any]]]:
    eta = _semantic_eta(dataset)
    panels = []
    selections = []
    xis = None
    max_crosscheck = 0.0
    for target in indentations:
        stored, selection, current_xis, case_names = _selected_field(
            dataset, mesh, target, "G_tangent"
        )
        panels.append(stored)
        selections.extend(selection)
        xis = current_xis if xis is None else xis
        for location_index, case in enumerate(case_names):
            case_index = dataset.case_index(case)
            delta = dataset.canonical_field("delta_n")[case_index]
            normal = dataset.canonical_field("u_normal")[case_index]
            independent = independent_tangent_gain(delta, normal)
            selected_independent = dataset.select_case_field(
                case, "G_tangent", target
            )
            # Selection metadata picks the same step/bracket. Apply it to the
            # independently differentiated array.
            if selected_independent.exact:
                comparison = independent[selected_independent.lower_step_index]
            else:
                weight = selected_independent.interpolation_weight
                comparison = (
                    (1.0 - weight) * independent[selected_independent.lower_step_index]
                    + weight * independent[selected_independent.upper_step_index]
                )
            max_crosscheck = max(
                max_crosscheck,
                float(np.max(np.abs(comparison - stored[location_index]))),
            )
    vlim = float(np.max(np.abs(panels)))
    norm = colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    image = None
    for axis, target, values in zip(axes.ravel(), indentations, panels):
        image = _heatmap_panel(
            axis,
            values,
            np.asarray(xis),
            eta,
            dataset.side_order,
            norm,
            cmap="BrBG",
            title=rf"$\delta={target:.2f}$ mm",
        )[0]
    figure.suptitle("Tangent transfer gain — stored Phase 4K finite differences")
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), label=r"$\partial u_n/\partial\delta$ [–]")
    outputs = _save_figure(
        figure, "tangent_transfer_gain", figures_dir, formats, dpi
    )
    rows = []
    for target, values in zip(indentations, panels):
        for location_index, xi in enumerate(xis):
            for side in ("right", "left"):
                side_index = dataset.side_index(side)
                for sample, (eta_value, gain) in enumerate(
                    zip(eta[side_index], values[location_index, side_index])
                ):
                    rows.append([mesh, target, xi, side, sample, eta_value, gain])
    metrics = {
        "stored_vs_independent_max_abs": max_crosscheck,
        "scheme": "np.gradient: centered interior, one-sided endpoints",
        "smoothing": False,
    }
    return outputs, rows, metrics, [
        {
            "name": "tangent_transfer_gain",
            "field": "G_tangent",
            "field_kind": "stored derived field, independently cross-checked",
            "units": "dimensionless",
            "finite_difference": "centered interior and one-sided endpoints",
            "smoothing": False,
            "color_limits": [-vlim, vlim],
            "input_selections": selections,
        }
    ]


def _medium_fine_figure(
    dataset: CODTMDataset,
    target: float,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[dict[str, Any]], list[list[Any]], dict[str, Any], list[dict[str, Any]]]:
    medium_map = _case_by_xi(dataset, "medium")
    fine_map = _case_by_xi(dataset, "fine")
    common_xi = [
        xi for xi in (0.20, 0.50, 0.80) if round(xi, 12) in medium_map and round(xi, 12) in fine_map
    ]
    eta = _semantic_eta(dataset)
    figure, axes = plt.subplots(
        len(common_xi), 2, figsize=(10.0, 2.8 * len(common_xi)), squeeze=False
    )
    rows = []
    metrics = []
    selections = []
    for row_index, xi in enumerate(common_xi):
        medium_selection = dataset.select_case_field(
            medium_map[round(xi, 12)], "u_normal", target
        )
        fine_selection = dataset.select_case_field(
            fine_map[round(xi, 12)], "u_normal", target
        )
        selections.extend(
            [
                {"case": medium_map[round(xi, 12)], "mesh": "medium", "xi_cmd": xi, **medium_selection.metadata()},
                {"case": fine_map[round(xi, 12)], "mesh": "fine", "xi_cmd": xi, **fine_selection.metadata()},
            ]
        )
        fine_common = common_eta_profiles(fine_selection.values, eta, eta)
        comparison = profile_comparison_metrics(medium_selection.values, fine_common)
        metrics.append({"xi_cmd": xi, **comparison})
        residual = medium_selection.values - fine_common
        for side in ("right", "left"):
            side_index = dataset.side_index(side)
            color = "C0" if side == "right" else "C1"
            axes[row_index, 0].plot(
                eta[side_index],
                medium_selection.values[side_index],
                color=color,
                label=f"medium {side}",
            )
            axes[row_index, 0].plot(
                eta[side_index],
                fine_common[side_index],
                color=color,
                ls="--",
                label=f"fine {side}",
            )
            axes[row_index, 1].plot(
                eta[side_index], residual[side_index], color=color, label=side
            )
            for sample, eta_value in enumerate(eta[side_index]):
                rows.append(
                    [
                        target,
                        xi,
                        side,
                        sample,
                        eta_value,
                        medium_selection.values[side_index, sample],
                        fine_common[side_index, sample],
                        residual[side_index, sample],
                        comparison["relative_l2"],
                        comparison["max_abs_mm"],
                        comparison["shape_correlation"],
                    ]
                )
        axes[row_index, 0].set_title(rf"$\xi={xi:.2f}$ profiles")
        axes[row_index, 1].set_title(
            f"residual; rel L2={100*comparison['relative_l2']:.3f}%, "
            f"corr={comparison['shape_correlation']:.6f}"
        )
        for axis in axes[row_index]:
            axis.axhline(0.0, color="0.6", lw=0.7)
            axis.set_xlabel(r"$\eta$")
            axis.grid(alpha=0.2)
        axes[row_index, 0].set_ylabel(r"$u_n$ [mm]")
        axes[row_index, 1].set_ylabel("medium − fine [mm]")
    if common_xi:
        axes[0, 0].legend(frameon=False, ncol=2)
        axes[0, 1].legend(frameon=False)
    figure.suptitle("Medium/fine CODTM profile comparison — status remains PROVISIONAL")
    figure.tight_layout()
    outputs = _save_figure(
        figure, "medium_fine_profiles_delta_1p50mm", figures_dir, formats, dpi
    )
    summary = {
        "status": "PROVISIONAL",
        "comparisons": metrics,
        "reported_range_reproduced": (
            min(item["relative_l2"] for item in metrics) >= 0.0024
            and max(item["relative_l2"] for item in metrics) <= 0.0068
            and min(item["shape_correlation"] for item in metrics) > 0.99996
        ),
        "note": "No post-hoc CODTM acceptance threshold was introduced.",
    }
    return outputs, rows, summary, [
        {
            "name": "medium_fine_profiles_delta_1p50mm",
            "field": "u_normal",
            "field_kind": "raw mesh comparison",
            "units": "mm",
            "status": "PROVISIONAL",
            "eta_mapping": "sidewise common eta; no xi interpolation",
            "input_selections": selections,
        }
    ]


def _overview(
    dataset: CODTMDataset,
    mesh: str,
    target: float,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    field, selections, xis, case_names = _selected_field(
        dataset, mesh, target, "u_normal"
    )
    eta = _semantic_eta(dataset)
    distance = location_distance_matrix(field, eta)
    vlim = float(np.max(np.abs(field)))
    figure = plt.figure(figsize=(12.0, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    schematic = figure.add_subplot(grid[0, 0])
    representative = min(range(len(xis)), key=lambda index: abs(xis[index] - 0.5))
    case = case_names[representative]
    for side, color in (("right", "C0"), ("left", "C1")):
        reference = dataset.reference_xy[(case, side)]
        schematic.plot(reference[:, 0], reference[:, 1], color=color, lw=2, label=f"{side} chain")
        schematic.scatter(reference[[0, -1], 0], reference[[0, -1], 1], color=color, s=18)
    schematic.set_aspect("equal", adjustable="box")
    schematic.set_xlabel("x [mm]")
    schematic.set_ylabel("y [mm]")
    schematic.set_title("A  Semantic observation sidewalls")
    schematic.legend(frameon=False)
    schematic.text(
        0.5,
        0.04,
        "central contact-facing region unsampled",
        transform=schematic.transAxes,
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "0.7"},
    )
    heatmap = figure.add_subplot(grid[0, 1])
    norm = colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    image = _heatmap_panel(
        heatmap,
        field,
        xis,
        eta,
        dataset.side_order,
        norm,
        cmap="RdBu_r",
        title=rf"B  CODTM at $\delta={target:.2f}$ mm",
    )[0]
    figure.colorbar(image, ax=heatmap, label=r"$u_n$ [mm]")
    profiles = figure.add_subplot(grid[1, 0])
    for location_index, xi in enumerate(xis):
        for side in ("right", "left"):
            side_index = dataset.side_index(side)
            profiles.plot(
                eta[side_index],
                field[location_index, side_index],
                color=_xi_color(float(xi)),
                ls="-" if side == "right" else "--",
                lw=1.4,
                label=rf"$\xi={xi:.2f}$" if side == "right" else None,
            )
    profiles.set_xlabel(r"$\eta$ (solid: right, dashed: left)")
    profiles.set_ylabel(r"$u_n$ [mm]")
    profiles.set_title("C  Independent sidewall profiles")
    profiles.legend(frameon=False, ncol=2)
    profiles.grid(alpha=0.2)
    matrix_axis = figure.add_subplot(grid[1, 1])
    matrix_image = matrix_axis.imshow(distance, origin="lower", cmap="magma", vmin=0.0)
    for i in range(len(xis)):
        for j in range(len(xis)):
            matrix_axis.text(j, i, f"{distance[i,j]:.3f}", ha="center", va="center", fontsize=6.5)
    matrix_axis.set_xticks(range(len(xis)), [f"{xi:.2f}" for xi in xis], rotation=45)
    matrix_axis.set_yticks(range(len(xis)), [f"{xi:.2f}" for xi in xis])
    matrix_axis.set_xlabel(r"$\xi_j$")
    matrix_axis.set_ylabel(r"$\xi_i$")
    matrix_axis.set_title("D  Location distance [mm]")
    figure.colorbar(matrix_image, ax=matrix_axis, label=r"$D_{ij}$ [mm]")
    figure.suptitle("Contact-to-Observation Deformation Transfer Map (CODTM)")
    figure.text(
        0.5,
        -0.015,
        "Outward u_normal is positive. Left/right are independent material chains; "
        "the center gap is unsampled. ξ is commanded, and heatmap rows are discrete solves.",
        ha="center",
    )
    outputs = _save_figure(figure, "codtm_overview", figures_dir, formats, dpi)
    return outputs, [
        {
            "name": "codtm_overview",
            "field": "u_normal",
            "field_kind": "composite raw field and derived distance",
            "units": {"displacement": "mm", "distance": "mm"},
            "input_selections": selections,
            "scientific_scope": (
                "mechanical transfer visualization only; no optical observability claim"
            ),
        }
    ]


def _attach_sources(
    manifest_entries: list[dict[str, Any]],
    source_mapping: Mapping[str, Sequence[str]],
    outputs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_stem: dict[str, list[dict[str, Any]]] = {}
    for output in outputs:
        by_stem.setdefault(Path(output["path"]).stem, []).append(output)
    result = []
    for entry in manifest_entries:
        current = dict(entry)
        name = str(current["name"])
        selections = current.get("input_selections", [])
        current["input_artifacts"] = [
            "codtm_arrays.npz",
            "codtm_long.csv",
            "map_metadata.json",
            "summary.json",
            "validation.json",
            "case_summary.csv",
            "source_trace.json",
        ]
        current["mesh"] = sorted(
            {
                str(selection["mesh"])
                for selection in selections
                if isinstance(selection, Mapping) and "mesh" in selection
            }
        )
        current["cases"] = sorted(
            {
                str(selection["case"])
                for selection in selections
                if isinstance(selection, Mapping) and "case" in selection
            }
        )
        current["indentations_mm"] = sorted(
            {
                float(selection["target_mm"])
                for selection in selections
                if isinstance(selection, Mapping) and "target_mm" in selection
            }
        )
        current["validity_mask"] = (
            "Phase 4K valid_mask must be true for exact or both bracketing steps; "
            "all selected displacement values must be finite"
        )
        current["source_data"] = list(source_mapping[name])
        current["outputs"] = by_stem.get(name, [])
        result.append(current)
    return result


def run_visualization(
    input_dir: Path,
    output_dir: Path,
    *,
    mesh: str = "medium",
    indentations: Sequence[float] = DEFAULT_INDENTATIONS,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    force: bool = False,
    diagnostics: bool = True,
) -> dict[str, Any]:
    """Run the full deterministic Phase 4K-Viz pipeline."""
    if mesh not in {"medium", "fine"}:
        raise CODTMVisualizationError("--mesh must be medium or fine")
    if not indentations or any(value <= 0.0 for value in indentations):
        raise CODTMVisualizationError("indentations must be positive")
    if set(formats) - {"png", "pdf"}:
        raise CODTMVisualizationError("formats must be png and/or pdf")
    if dpi < 72:
        raise CODTMVisualizationError("dpi must be at least 72")
    input_root = input_dir.resolve()
    output_root = output_dir.resolve()
    before = input_checksums(input_root)
    if output_root.exists():
        if not force:
            raise CODTMVisualizationError(
                f"output directory exists; use --force: {output_root}"
            )
        shutil.rmtree(output_root)
    figure_data_dir = output_root / "figure_data"
    figures_dir = output_root / "figures"
    figure_data_dir.mkdir(parents=True)
    figures_dir.mkdir()
    _style()
    dataset, audit = load_codtm_dataset(input_root)
    all_outputs: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    outputs, spatial_rows, entries = _spatial_atlas(
        dataset, mesh, indentations, figures_dir, formats, dpi
    )
    all_outputs += outputs
    manifest_entries += entries
    _write_csv(
        figure_data_dir / "spatial_atlas.csv",
        ("mesh", "delta_mm", "xi_cmd", "side", "sample", "eta", "display_zeta", "u_normal_mm"),
        spatial_rows,
    )

    final_target = max(indentations)
    outputs, profile_rows, physical_rows, entries = _profiles_and_physical(
        dataset, mesh, final_target, figures_dir, formats, dpi
    )
    all_outputs += outputs
    manifest_entries += entries
    _write_csv(
        figure_data_dir / "profiles_1p50mm.csv",
        ("mesh", "delta_mm", "xi_cmd", "side", "sample", "eta", "u_normal_mm", "u_x_mm", "u_y_mm"),
        profile_rows,
    )
    _write_csv(
        figure_data_dir / "physical_sidewalls_1p50mm.csv",
        (
            "mesh", "delta_mm", "xi_cmd", "side", "sample", "eta",
            "X0_x_mm", "X0_y_mm", "u_x_mm", "u_y_mm", "deformed_x_mm", "deformed_y_mm",
        ),
        physical_rows,
    )

    outputs, secant_rows, entries = _secant_atlas(
        dataset, mesh, indentations, figures_dir, formats, dpi
    )
    all_outputs += outputs
    manifest_entries += entries
    _write_csv(
        figure_data_dir / "secant_gain.csv",
        ("mesh", "delta_mm", "xi_cmd", "side", "sample", "eta", "G_secant"),
        secant_rows,
    )

    outputs, distance_rows, shape_rows, distance_metrics, entries = _distance_figures(
        dataset, mesh, indentations, figures_dir, formats, dpi
    )
    all_outputs += outputs
    manifest_entries += entries
    _write_csv(
        figure_data_dir / "distance_matrices.csv",
        ("mesh", "delta_mm", "xi_i", "xi_j", "distance_mm"),
        distance_rows,
    )
    _write_csv(
        figure_data_dir / "shape_distance_1p50mm.csv",
        ("mesh", "delta_mm", "xi_i", "xi_j", "shape_distance"),
        shape_rows,
    )

    metrics: dict[str, Any] = {
        "phase": "4K-Viz",
        "distance": distance_metrics,
        "data_ingestion": "PASS",
        "coordinate_semantics": "PASS",
        "static_visualization_pipeline": "PASS",
    }
    if diagnostics:
        outputs, mirror_rows, mirror_summary, entries = _mirror_figure(
            dataset, mesh, final_target, figures_dir, formats, dpi
        )
        all_outputs += outputs
        manifest_entries += entries
        _write_csv(
            figure_data_dir / "mirror_symmetry.csv",
            (
                "mesh", "delta_mm", "xi", "mirror_xi", "side", "sample", "eta",
                "original_u_normal_mm", "mirrored_u_normal_mm", "residual_mm",
                "absolute_l2_mm", "relative_l2", "max_abs_mm",
            ),
            mirror_rows,
        )
        metrics["mirror_symmetry"] = mirror_summary

        outputs, tangent_rows, tangent_summary, entries = _tangent_figure(
            dataset, mesh, indentations, figures_dir, formats, dpi
        )
        all_outputs += outputs
        manifest_entries += entries
        _write_csv(
            figure_data_dir / "tangent_gain.csv",
            ("mesh", "delta_mm", "xi_cmd", "side", "sample", "eta", "G_tangent"),
            tangent_rows,
        )
        metrics["tangent_gain"] = tangent_summary

        outputs, mesh_rows, mesh_summary, entries = _medium_fine_figure(
            dataset, final_target, figures_dir, formats, dpi
        )
        all_outputs += outputs
        manifest_entries += entries
        _write_csv(
            figure_data_dir / "medium_fine_profiles.csv",
            (
                "delta_mm", "xi_cmd", "side", "sample", "eta",
                "medium_u_normal_mm", "fine_u_normal_mm", "residual_mm",
                "relative_l2", "max_abs_mm", "shape_correlation",
            ),
            mesh_rows,
        )
        metrics["medium_fine"] = mesh_summary

    overview_outputs, entries = _overview(
        dataset, mesh, final_target, figures_dir, formats, dpi
    )
    all_outputs += overview_outputs
    manifest_entries += entries

    source_mapping = {
        "codtm_spatial_atlas": ["figure_data/spatial_atlas.csv"],
        "codtm_profiles_delta_1p50mm": ["figure_data/profiles_1p50mm.csv"],
        "sidewall_deformation_delta_1p50mm": ["figure_data/physical_sidewalls_1p50mm.csv"],
        "codtm_secant_gain_atlas": ["figure_data/secant_gain.csv"],
        "location_distance_matrices": ["figure_data/distance_matrices.csv"],
        "shape_distance_delta_1p50mm": ["figure_data/shape_distance_1p50mm.csv"],
        "mirror_symmetry_delta_1p50mm": ["figure_data/mirror_symmetry.csv"],
        "tangent_transfer_gain": ["figure_data/tangent_gain.csv"],
        "medium_fine_profiles_delta_1p50mm": ["figure_data/medium_fine_profiles.csv"],
        "codtm_overview": [
            "figure_data/physical_sidewalls_1p50mm.csv",
            "figure_data/spatial_atlas.csv",
            "figure_data/profiles_1p50mm.csv",
            "figure_data/distance_matrices.csv",
        ],
    }
    manifest = _attach_sources(manifest_entries, source_mapping, all_outputs)
    after = input_checksums(input_root)
    audit["input_checksums_after_sha256"] = after
    audit["canonical_inputs_unchanged"] = before == after
    audit["descriptor_verified_count"] = int(descriptor_verified_mask(dataset).sum())
    audit["descriptor_total_count"] = int(descriptor_verified_mask(dataset).size)
    audit["descriptor_unverified_policy"] = (
        "NaN remains unverified; no zero-fill, location interpolation, or mirror reconstruction"
    )
    audit["status"] = "PASS" if before == after else "FAIL"
    metrics["metric_reproduction"] = (
        "PASS"
        if 0.264 <= distance_metrics["raw_distance_1p50mm_offdiagonal_min_mm"] <= 0.267
        and 0.980 <= distance_metrics["raw_distance_1p50mm_offdiagonal_max_mm"] <= 0.984
        and 0.551 <= distance_metrics["shape_distance_1p50mm_offdiagonal_min"] <= 0.554
        and 1.967 <= distance_metrics["shape_distance_1p50mm_offdiagonal_max"] <= 1.970
        else "FAIL"
    ) if mesh == "medium" and abs(final_target - 1.5) < 1e-12 else "NOT_APPLICABLE"
    publication_ready = (
        audit["status"] == "PASS"
        and metrics["metric_reproduction"] in {"PASS", "NOT_APPLICABLE"}
        and all(output["bytes"] > 0 for output in all_outputs)
    )
    metrics["publication_status"] = "READY" if publication_ready else "NEEDS REVISION"
    metadata = {
        "phase": "4K-Viz",
        "status": "PASS" if publication_ready else "FAIL",
        "input_directory": str(input_root),
        "output_directory": str(output_root),
        "mesh": mesh,
        "indentations_mm": [float(value) for value in indentations],
        "formats": list(formats),
        "dpi": dpi,
        "diagnostics": diagnostics,
        "coordinate_contract": {
            "primary": "(side, eta)",
            "eta": "0 bonded endpoint; 1 crownward observation endpoint",
            "zeta_right": "eta - 1",
            "zeta_left": "1 - eta",
            "display_gap_width": DISPLAY_GAP,
            "display_gap_note": (
                "visual separation only; zeta=0- and zeta=0+ are distinct material points"
            ),
        },
        "scientific_scope": (
            "static mechanical CODTM visualization; no optical/noise observability claim"
        ),
    }
    _strict_write_json(output_root / "visualization_metadata.json", metadata)
    _strict_write_json(output_root / "data_audit.json", audit)
    _strict_write_json(output_root / "metrics_summary.json", metrics)
    _strict_write_json(
        output_root / "plot_manifest.json",
        {"phase": "4K-Viz", "figures": manifest},
    )
    csv_audits = []
    for path in sorted(figure_data_dir.glob("*.csv")):
        csv_audits.append(
            finite_csv_audit(path, nonnumeric={"mesh", "side"})
        )
    result = {
        "status": metadata["status"],
        "output_directory": str(output_root),
        "figure_count": len(manifest),
        "rendered_file_count": len(all_outputs),
        "source_csv_count": len(csv_audits),
        "metrics": metrics,
        "data_audit": audit,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh", choices=("medium", "fine"), default="medium")
    parser.add_argument(
        "--indentations",
        nargs="+",
        type=float,
        default=list(DEFAULT_INDENTATIONS),
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"]
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_visualization(
            args.input_dir,
            args.output_dir,
            mesh=args.mesh,
            indentations=tuple(args.indentations),
            formats=tuple(args.formats),
            dpi=args.dpi,
            force=args.force,
            diagnostics=not args.no_diagnostics,
        )
    except CODTMVisualizationError as exc:
        print(f"Phase 4K-Viz FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
