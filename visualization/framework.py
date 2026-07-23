"""Declarative figure specifications, builders, public API, and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from visualization.data import (
    FRAMEWORK_VERSION,
    ScientificFigureError,
    VisualizationDataset,
    merge_visualization_datasets,
)
from visualization.adapters.normal_indentation import (
    load_normal_indentation_visualization_dataset,
)
from visualization.adapters.phase4k_dataset import (
    load_phase4k_visualization_dataset,
)
from visualization.export import FigureExporter, RenderedFigure, SourceTable
from visualization.panels.geometry import (
    ContactInputAnnotation,
    DisplacementVectorPanel,
    MeshPanel,
    NodalDisplacementMagnitudePanel,
    ObservationBoundaryOverlay,
)
from visualization.panels.transfer import (
    FigureComposer,
    LocationDistanceMatrixPanel,
    MetricSummaryPanel,
    PanelLabelManager,
    SharedColorbar,
    TransferMapPanel,
)
from visualization.theme import FigureTheme, ScalePolicy
from visualization.transforms import (
    SelectedTransferState,
    raw_distance_metrics,
    select_transfer_state,
    symmetric_limits,
    transfer_summary_metrics,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FigureSpec:
    """Validated declarative contract for one rendered figure."""

    kind: str
    title: str
    basename: str
    designs: tuple[str, ...]
    mesh_id: str
    quantity: str
    contact_locations: tuple[float, ...]
    contact_coordinate: str
    indentation_mm: float
    interpolation_enabled: bool
    color_scale_mode: str
    color_center: float
    panels: Mapping[str, bool]
    theme: str
    deformation_scale: float
    arrow_scale: float
    arrow_minimum_mm: float
    maximum_arrows_per_panel: int
    export_formats: tuple[str, ...]
    raster_dpi: int
    output_directory: str
    raw_spec: Mapping[str, Any]
    spec_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "transfer_map_comparison",
            "displacement_vector_atlas",
        }:
            raise ScientificFigureError(f"unsupported figure kind {self.kind!r}")
        if not self.designs or not self.contact_locations:
            raise ScientificFigureError("figure designs/contact locations are empty")
        if self.indentation_mm <= 0.0:
            raise ScientificFigureError("indentation must be positive")
        if self.kind == "transfer_map_comparison":
            if (
                self.color_scale_mode != "shared_symmetric"
                or self.color_center != 0.0
            ):
                raise ScientificFigureError(
                    "signed reference figures require a shared zero-centered scale"
                )
            if self.contact_coordinate != "normalized_arc":
                raise ScientificFigureError(
                    "transfer-map contact coordinate must be normalized_arc"
                )
        elif self.color_scale_mode not in {
            "shared_symmetric",
            "shared_sequential",
        }:
            raise ScientificFigureError(
                "displacement atlas color scale must be shared_symmetric "
                "or shared_sequential"
            )
        if self.contact_coordinate not in {"normalized_arc", "surface_x_mm"}:
            raise ScientificFigureError("unsupported contact coordinate")
        if set(self.export_formats) - {"png", "pdf"}:
            raise ScientificFigureError("export formats must be png/pdf")
        if self.maximum_arrows_per_panel < 1 or self.raster_dpi < 72:
            raise ScientificFigureError("arrow count or raster DPI is invalid")


def _strict_json_text(text: str, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard constant {constant}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ScientificFigureError(
            f"{source} must be strict JSON or JSON-compatible YAML"
        ) from exc
    if not isinstance(value, dict):
        raise ScientificFigureError("FigureSpec root must be an object")
    return value


def _parse_figure_spec(
    value: Mapping[str, Any],
    *,
    spec_path: str | None,
) -> FigureSpec:
    figure = value.get("figure")
    if not isinstance(figure, Mapping):
        raise ScientificFigureError("FigureSpec requires a figure object")
    export = figure.get("export", {})
    scale = figure.get("scale", {})
    interpolation = figure.get("interpolation", {})
    color_scale = figure.get("color_scale", {})
    arrows = figure.get("arrows", {})
    return FigureSpec(
        kind=str(figure["kind"]),
        title=str(figure["title"]),
        basename=str(figure.get("basename", figure["kind"])),
        designs=tuple(str(item) for item in figure["designs"]),
        mesh_id=str(figure.get("mesh", "medium")),
        quantity=str(figure.get("quantity", "raw_displacement")),
        contact_locations=tuple(float(item) for item in figure["contact_locations"]),
        contact_coordinate=str(
            figure.get("contact_coordinate", "normalized_arc")
        ),
        indentation_mm=float(figure["indentation_mm"]),
        interpolation_enabled=bool(interpolation.get("enabled", False)),
        color_scale_mode=str(color_scale.get("mode", "shared_symmetric")),
        color_center=float(color_scale.get("center", 0.0)),
        panels={
            str(name): bool(enabled)
            for name, enabled in figure.get("panels", {}).items()
        },
        theme=str(figure.get("theme", "journal")),
        deformation_scale=float(scale.get("deformation", 1.0)),
        arrow_scale=float(scale.get("arrows", 1.0)),
        arrow_minimum_mm=float(arrows.get("minimum_magnitude_mm", 0.0)),
        maximum_arrows_per_panel=int(arrows.get("maximum_per_panel", 18)),
        export_formats=tuple(str(item) for item in export.get("formats", ["png", "pdf"])),
        raster_dpi=int(export.get("dpi", 300)),
        output_directory=str(export["output_directory"]),
        raw_spec=dict(value),
        spec_path=spec_path,
    )


def load_figure_spec(path_or_mapping: str | Path | Mapping[str, Any]) -> FigureSpec:
    """Load strict JSON, including JSON syntax stored in a YAML 1.2 file."""
    if isinstance(path_or_mapping, Mapping):
        return _parse_figure_spec(path_or_mapping, spec_path=None)
    path = Path(path_or_mapping).resolve()
    value = _strict_json_text(path.read_text(encoding="utf-8"), str(path))
    return _parse_figure_spec(value, spec_path=str(path))


def _resolve_repository_path(value: str, spec: FigureSpec) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPOSITORY_ROOT / path


def load_visualization_dataset(spec: FigureSpec) -> VisualizationDataset:
    """Load one or more declared designs through repository data adapters."""
    dataset_specs = spec.raw_spec.get("datasets")
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise ScientificFigureError("FigureSpec requires a nonempty datasets list")
    loaded = []
    declared_designs = set()
    for item in dataset_specs:
        if not isinstance(item, Mapping):
            raise ScientificFigureError("dataset entry must be an object")
        adapter = str(item.get("adapter", ""))
        design_id = str(item["design_id"])
        if design_id in declared_designs:
            raise ScientificFigureError(f"duplicate design ID {design_id}")
        declared_designs.add(design_id)
        input_dir = _resolve_repository_path(str(item["input_dir"]), spec)
        if adapter == "phase4k":
            loaded.append(
                load_phase4k_visualization_dataset(
                    input_dir,
                    design_id=design_id,
                    mesh_ids=(spec.mesh_id,),
                )
            )
        elif adapter == "normal_indentation_full_field":
            loaded.append(
                load_normal_indentation_visualization_dataset(
                    input_dir,
                    design_id=design_id,
                    mesh_id=spec.mesh_id,
                )
            )
        else:
            raise ScientificFigureError(f"unsupported dataset adapter {adapter!r}")
    missing = set(spec.designs) - declared_designs
    if missing:
        raise ScientificFigureError(
            f"figure references undeclared designs {sorted(missing)}"
        )
    return merge_visualization_datasets(loaded)


def _selected_states(
    dataset: VisualizationDataset,
    spec: FigureSpec,
    design_id: str,
) -> list[SelectedTransferState]:
    return [
        select_transfer_state(
            dataset,
            design_id=design_id,
            mesh_id=spec.mesh_id,
            xi=xi,
            delta_mm=spec.indentation_mm,
            quantity=spec.quantity,
            interpolation_enabled=spec.interpolation_enabled,
        )
        for xi in spec.contact_locations
    ]


def _quantity_label(quantity: str, units: str) -> str:
    labels = {
        "raw_displacement": r"$u_\mathrm{normal}$",
        "secant_gain": r"$G_\mathrm{secant}=u_n/\delta$",
        "force_compliance": r"$C_\mathrm{force}=u_n/F$",
        "tangent_gain": r"$G_\mathrm{tangent}=\partial u_n/\partial\delta$",
    }
    return f"{labels[quantity]} [{units}]"


class TransferMapComparisonFigure:
    """Build a single- or multi-design geometry-conditioned CODTM figure."""

    def build(
        self,
        dataset: VisualizationDataset,
        spec: FigureSpec,
        theme: FigureTheme,
    ) -> RenderedFigure:
        states_by_design = {
            design_id: _selected_states(dataset, spec, design_id)
            for design_id in spec.designs
        }
        color_limits = symmetric_limits(states_by_design)
        signed_norm = colors.TwoSlopeNorm(
            vmin=color_limits[0], vcenter=0.0, vmax=color_limits[1]
        )
        distances = {
            design_id: raw_distance_metrics(states)
            for design_id, states in states_by_design.items()
        }
        distance_maximum = max(float(matrix.max()) for matrix in distances.values())
        distance_norm = colors.Normalize(vmin=0.0, vmax=distance_maximum)
        if len(spec.designs) > 2:
            raise ScientificFigureError(
                "reference transfer-map builder supports one or two designs"
            )
        if spec.panels.get("optimization_history", False):
            raise ScientificFigureError(
                "optimization history was requested but no history artifact is declared"
            )
        panel_plan: list[tuple[str, str | None]] = []
        if spec.panels.get("geometry_context", True):
            panel_plan.extend(("geometry", design_id) for design_id in spec.designs)
        if spec.panels.get("transfer_maps", True):
            panel_plan.extend(("transfer", design_id) for design_id in spec.designs)
        if spec.panels.get("distance_matrices", True):
            panel_plan.extend(("distance", design_id) for design_id in spec.designs)
        if spec.panels.get("metric_summary", True):
            panel_plan.append(("summary", None))
        if not panel_plan:
            raise ScientificFigureError("FigureSpec disables every transfer-map panel")
        columns = 2 if len(panel_plan) <= 4 else 3
        rows = int(np.ceil(len(panel_plan) / columns))
        composer = FigureComposer()
        figure, axes = composer.create(
            rows,
            columns,
            figsize=(5.3 * columns, 3.9 * rows),
            title=spec.title,
        )
        all_axes = list(axes.ravel())
        active_axes = all_axes[: len(panel_plan)]
        for unused in all_axes[len(panel_plan) :]:
            unused.set_visible(False)
        panel_metadata: list[Mapping[str, Any]] = []
        map_axes: list[plt.Axes] = []
        distance_axes: list[plt.Axes] = []
        map_image = None
        distance_image = None
        for axis, (panel_kind, design_id) in zip(active_axes, panel_plan):
            if panel_kind == "geometry":
                assert design_id is not None
                mesh = dataset.mesh(design_id, spec.mesh_id)
                chains = {
                    side: dataset.chain(design_id, spec.mesh_id, side)
                    for side in ("right", "left")
                }
                panel_metadata.append(MeshPanel().render(axis, mesh, theme))
                panel_metadata.append(
                    ObservationBoundaryOverlay().render(axis, chains, theme)
                )
                axis.set_title(
                    f"{design_id}: actual T3 geometry and sampled sidewalls"
                )
            elif panel_kind == "transfer":
                assert design_id is not None
                map_meta = TransferMapPanel().render(
                    axis, states_by_design[design_id], theme, signed_norm
                )
                map_image = map_meta.pop("image")
                panel_metadata.append(map_meta)
                map_axes.append(axis)
                axis.set_title(f"{design_id}: discrete mechanical transfer map")
            elif panel_kind == "distance":
                assert design_id is not None
                matrix_meta = LocationDistanceMatrixPanel().render(
                    axis,
                    distances[design_id],
                    spec.contact_locations,
                    theme,
                    distance_norm,
                )
                distance_image = matrix_meta.pop("image")
                panel_metadata.append(matrix_meta)
                distance_axes.append(axis)
                axis.set_title(f"{design_id}: pairwise signature distance")
            else:
                metrics = {
                    item_design: transfer_summary_metrics(states)
                    for item_design, states in states_by_design.items()
                }
                panel_metadata.append(MetricSummaryPanel().render(axis, metrics))
                axis.set_title("Descriptive mechanical metrics")
        first_design = spec.designs[0]
        if map_axes:
            assert map_image is not None
            panel_metadata.append(
                SharedColorbar().add(
                    figure,
                    map_image,
                    map_axes,
                    label=_quantity_label(
                        spec.quantity, states_by_design[first_design][0].units
                    ),
                )
            )
        if distance_axes:
            assert distance_image is not None
            panel_metadata.append(
                SharedColorbar().add(
                    figure,
                    distance_image,
                    distance_axes,
                    label=(
                        "signature distance "
                        f"[{states_by_design[first_design][0].units}]"
                    ),
                )
            )
        panel_metadata.append(PanelLabelManager().apply(active_axes, theme))
        figure.text(
            0.5,
            -0.01,
            "Transfer map is a geometry-conditioned prescribed-indentation response, "
            "not a classical s-domain transfer function. Center region is unsampled.",
            ha="center",
        )
        transfer_rows = []
        distance_rows = []
        metric_rows = []
        cases = []
        interpolation_records = []
        for design_id, states in states_by_design.items():
            summary = transfer_summary_metrics(states)
            for name, value in summary.items():
                if value is not None:
                    metric_rows.append((design_id, name, value))
            for state in states:
                cases.append(state.case_id)
                interpolation_records.append(
                    {"design_id": design_id, "xi": state.xi, **state.selection_metadata}
                )
                for side in ("right", "left"):
                    for sample, (eta, value) in enumerate(
                        zip(state.eta_by_side[side], state.values_by_side[side])
                    ):
                        transfer_rows.append(
                            (
                                design_id,
                                spec.mesh_id,
                                state.case_id,
                                state.xi,
                                state.delta_mm,
                                side,
                                sample,
                                eta,
                                value,
                            )
                        )
            matrix = distances[design_id]
            for row, xi_i in enumerate(spec.contact_locations):
                for column, xi_j in enumerate(spec.contact_locations):
                    distance_rows.append(
                        (design_id, xi_i, xi_j, matrix[row, column])
                    )
        units = states_by_design[spec.designs[0]][0].units
        return RenderedFigure(
            figure=figure,
            basename=spec.basename,
            figure_kind=spec.kind,
            represented_variable=spec.quantity,
            units=units,
            normalization=states_by_design[spec.designs[0]][0].normalization,
            design_ids=spec.designs,
            mesh_ids=(spec.mesh_id,),
            cases=tuple(sorted(set(cases))),
            xi_values=spec.contact_locations,
            indentation_values_mm=(spec.indentation_mm,),
            coordinate_convention=dataset.metadata["adapters"][0][
                "coordinate_convention"
            ],
            interpolation={
                "enabled": spec.interpolation_enabled,
                "xi_interpolation": False,
                "selections": interpolation_records,
            },
            validity={
                "displacement": "CODTM valid and finite",
                "descriptor": "separate; not used by transfer map",
            },
            scale_policy=ScalePolicy(
                deformation_scale=spec.deformation_scale,
                arrow_scale=spec.arrow_scale,
                arrow_minimum_mm=spec.arrow_minimum_mm,
                color_limits=color_limits,
            ),
            color_limits=color_limits,
            panel_metadata=tuple(panel_metadata),
            source_tables=(
                SourceTable(
                    "transfer_map.csv",
                    (
                        "design_id",
                        "mesh_id",
                        "case_id",
                        "xi",
                        "delta_mm",
                        "side",
                        "sample",
                        "eta",
                        "value",
                    ),
                    tuple(transfer_rows),
                    label_columns=("design_id", "mesh_id", "case_id", "side"),
                ),
                SourceTable(
                    "location_distance.csv",
                    ("design_id", "xi_i", "xi_j", "distance"),
                    tuple(distance_rows),
                    label_columns=("design_id",),
                ),
                SourceTable(
                    "metric_summary.csv",
                    ("design_id", "metric", "value"),
                    tuple(metric_rows),
                    label_columns=("design_id", "metric"),
                ),
            ),
            notes=(
                "Single-design mode is used when no optimized artifact is supplied.",
                "No optical observability or optimization claim is made.",
            ),
        )


class DisplacementVectorAtlasFigure:
    """Build an actual FEM field view or the legacy sidewall-only view."""

    def build(
        self,
        dataset: VisualizationDataset,
        spec: FigureSpec,
        theme: FigureTheme,
    ) -> RenderedFigure:
        if spec.contact_coordinate == "surface_x_mm":
            return self._build_full_field(dataset, spec, theme)
        return self._build_observation_chain(dataset, spec, theme)

    def _build_full_field(
        self,
        dataset: VisualizationDataset,
        spec: FigureSpec,
        theme: FigureTheme,
    ) -> RenderedFigure:
        if len(spec.designs) != 1:
            raise ScientificFigureError(
                "full-field displacement atlas requires exactly one design"
            )
        if spec.color_scale_mode != "shared_sequential":
            raise ScientificFigureError(
                "displacement-magnitude heatmap requires shared_sequential color"
            )
        design_id = spec.designs[0]
        mesh = dataset.mesh(design_id, spec.mesh_id)
        selected: list[tuple[Any, Any]] = []
        for surface_x_mm in spec.contact_locations:
            matches = [
                case
                for case in dataset.contact_cases
                if case.design_id == design_id
                and case.mesh_id == spec.mesh_id
                and case.surface_x_mm is not None
                and np.isclose(
                    case.surface_x_mm, surface_x_mm, rtol=0.0, atol=1.0e-12
                )
                and np.isclose(
                    case.delta_mm,
                    spec.indentation_mm,
                    rtol=0.0,
                    atol=1.0e-9,
                )
                and case.convergence_state == "converged"
            ]
            if len(matches) != 1:
                raise ScientificFigureError(
                    f"expected one converged full-field state at "
                    f"x={surface_x_mm:g} mm"
                )
            case = matches[0]
            field = dataset.displacement_state(
                design_id, case.case_id, case.step
            )
            if field.location_kind != "mesh_node" or not field.validity_mask.all():
                raise ScientificFigureError(
                    f"{case.case_id} is not a valid full mesh-node field"
                )
            if set(field.point_ids) != {str(node_id) for node_id in mesh.node_ids}:
                raise ScientificFigureError(
                    f"{case.case_id} displacement nodes do not match the mesh"
                )
            selected.append((case, field))

        maximum_magnitude = max(
            float(np.linalg.norm(field.nodal_displacement, axis=1).max())
            for _, field in selected
        )
        if not np.isfinite(maximum_magnitude) or maximum_magnitude <= 0.0:
            raise ScientificFigureError(
                "full-field displacement magnitude has no positive finite value"
            )
        color_limits = (0.0, maximum_magnitude)
        magnitude_norm = colors.Normalize(*color_limits)
        scale_policy = ScalePolicy(
            deformation_scale=spec.deformation_scale,
            arrow_scale=spec.arrow_scale,
            arrow_minimum_mm=spec.arrow_minimum_mm,
            color_limits=color_limits,
        )
        composer = FigureComposer()
        figure, axes = composer.create(
            1,
            len(selected),
            figsize=(4.5 * len(selected), 5.0),
            title=spec.title,
        )
        flat_axes = list(axes.ravel())
        panel_metadata: list[Mapping[str, Any]] = []
        vector_rows = []
        image = None
        for axis, (case, field) in zip(flat_axes, selected):
            vector_by_node_id = {
                int(point_id): field.nodal_displacement[index]
                for index, point_id in enumerate(field.point_ids)
            }
            displacement = np.asarray(
                [vector_by_node_id[node_id] for node_id in mesh.node_ids]
            )
            deformed_coordinates = (
                mesh.node_coordinates
                + scale_policy.deformation_scale * displacement
            )
            if spec.panels.get("fem_heatmap", True):
                heatmap_meta = NodalDisplacementMagnitudePanel().render(
                    axis,
                    mesh,
                    vector_by_node_id,
                    theme,
                    scale_policy,
                    magnitude_norm,
                )
                image = heatmap_meta.pop("image")
                panel_metadata.append(heatmap_meta)
            if spec.panels.get("displacement_vectors", True):
                vector_meta, arrow_indices = DisplacementVectorPanel().render(
                    axis,
                    deformed_coordinates,
                    displacement,
                    theme,
                    scale_policy,
                    maximum_arrows=spec.maximum_arrows_per_panel,
                    color="#111111",
                    anchor_configuration="deformed",
                )
                panel_metadata.append(vector_meta)
            else:
                arrow_indices = np.asarray([], dtype=int)
            if spec.panels.get("contact_annotation", True):
                reference_point = np.asarray(case.contact_point_mm, dtype=float)
                direction = np.asarray(case.indentation_direction, dtype=float)
                final_tangent_point = (
                    reference_point + case.delta_mm * direction
                )
                annotation = ContactInputAnnotation().render_at(
                    axis,
                    final_tangent_point,
                    direction,
                )
                annotation["reference_contact_point_mm"] = reference_point.tolist()
                annotation["display_point"] = (
                    "final prescribed indenter tangency point"
                )
                panel_metadata.append(annotation)
            reaction_text = (
                "reaction unavailable"
                if case.reaction_force_n is None
                else f"F={case.reaction_force_n:.4f} N"
            )
            axis.set_title(
                f"x={case.surface_x_mm:g} mm, "
                f"n_in=({case.indentation_direction[0]:+.3f}, "
                f"{case.indentation_direction[1]:+.3f})\n"
                f"δ={case.delta_mm:.2f} mm, {reaction_text}"
            )
            axis.text(
                0.02,
                0.02,
                f"heatmap geometry {scale_policy.deformation_scale:g}×; "
                f"vectors {scale_policy.arrow_scale:g}×",
                transform=axis.transAxes,
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
                zorder=20,
            )
            arrow_set = set(int(index) for index in arrow_indices)
            magnitude = np.linalg.norm(displacement, axis=1)
            for index, node_id in enumerate(mesh.node_ids):
                vector_rows.append(
                    (
                        design_id,
                        spec.mesh_id,
                        case.case_id,
                        case.surface_x_mm,
                        case.xi,
                        case.delta_mm,
                        node_id,
                        mesh.node_coordinates[index, 0],
                        mesh.node_coordinates[index, 1],
                        deformed_coordinates[index, 0],
                        deformed_coordinates[index, 1],
                        displacement[index, 0],
                        displacement[index, 1],
                        magnitude[index],
                        int(index in arrow_set),
                    )
                )
        if image is None:
            raise ScientificFigureError("full-field heatmap panel is disabled")
        panel_metadata.append(
            SharedColorbar().add(
                figure,
                image,
                flat_axes,
                label=r"displacement magnitude $|u|$ [mm]",
            )
        )
        panel_metadata.append(PanelLabelManager().apply(flat_axes, theme))
        figure.text(
            0.5,
            -0.01,
            "Heatmap: full-pad FEM |u| on the 1× deformed T3 mesh. "
            "Arrows: actual u=[u_x,u_y], anchored on that deformed mesh. "
            "Each indenter moves along the local inward curvature normal.",
            ha="center",
        )
        node_rows = tuple(
            (design_id, spec.mesh_id, node_id, point[0], point[1])
            for node_id, point in zip(mesh.node_ids, mesh.node_coordinates)
        )
        element_rows = tuple(
            (
                design_id,
                spec.mesh_id,
                element_id,
                int(connectivity[0]),
                int(connectivity[1]),
                int(connectivity[2]),
            )
            for element_id, connectivity in zip(
                mesh.element_ids, mesh.element_connectivity
            )
        )
        return RenderedFigure(
            figure=figure,
            basename=spec.basename,
            figure_kind=spec.kind,
            represented_variable=(
                "full-pad nodal displacement magnitude |u| and vector "
                "u=[u_x,u_y]"
            ),
            units="mm",
            normalization="none",
            design_ids=spec.designs,
            mesh_ids=(spec.mesh_id,),
            cases=tuple(case.case_id for case, _ in selected),
            xi_values=tuple(case.xi for case, _ in selected),
            surface_x_values_mm=tuple(
                float(case.surface_x_mm) for case, _ in selected
            ),
            indentation_values_mm=(spec.indentation_mm,),
            coordinate_convention=dataset.metadata["adapters"][0][
                "coordinate_convention"
            ],
            interpolation={
                "state_selection": "exact surface x and indentation",
                "state_interpolation": False,
                "heatmap": "linear T3 nodal interpolation",
                "vector_subsampling": "deterministic spatial binning",
            },
            validity={
                "displacement": "converged, finite, full pad mesh-node field",
                "field_nodes": "exact match to persisted pad mesh",
                "contact_direction": "local inward normal at each surface x",
            },
            scale_policy=scale_policy,
            color_limits=color_limits,
            panel_metadata=tuple(panel_metadata),
            source_tables=(
                SourceTable(
                    "mesh_nodes.csv",
                    ("design_id", "mesh_id", "node_id", "x_mm", "y_mm"),
                    node_rows,
                    label_columns=("design_id", "mesh_id"),
                ),
                SourceTable(
                    "mesh_elements.csv",
                    (
                        "design_id",
                        "mesh_id",
                        "element_id",
                        "node_1",
                        "node_2",
                        "node_3",
                    ),
                    element_rows,
                    label_columns=("design_id", "mesh_id"),
                ),
                SourceTable(
                    "displacement_vectors.csv",
                    (
                        "design_id",
                        "mesh_id",
                        "case_id",
                        "surface_x_mm",
                        "xi",
                        "delta_mm",
                        "node_id",
                        "X0_x_mm",
                        "X0_y_mm",
                        "x_deformed_mm",
                        "y_deformed_mm",
                        "u_x_mm",
                        "u_y_mm",
                        "u_magnitude_mm",
                        "arrow_selected",
                    ),
                    tuple(vector_rows),
                    label_columns=("design_id", "mesh_id", "case_id"),
                ),
            ),
            notes=(
                "All three panels are independent nonlinear FEM solves.",
                "Only the deformable pad field is shown; rigid carrier and indenter are excluded.",
                "The heatmap is displacement magnitude and uses one shared sequential scale.",
            ),
        )

    def _build_observation_chain(
        self,
        dataset: VisualizationDataset,
        spec: FigureSpec,
        theme: FigureTheme,
    ) -> RenderedFigure:
        if len(spec.designs) != 1:
            raise ScientificFigureError(
                "displacement vector atlas requires exactly one design"
            )
        design_id = spec.designs[0]
        raw_spec = FigureSpec(
            **{
                **spec.__dict__,
                "quantity": "raw_displacement",
            }
        )
        states = _selected_states(dataset, raw_spec, design_id)
        color_limits = symmetric_limits({design_id: states})
        signed_norm = colors.TwoSlopeNorm(
            vmin=color_limits[0], vcenter=0.0, vmax=color_limits[1]
        )
        scale_policy = ScalePolicy(
            deformation_scale=spec.deformation_scale,
            arrow_scale=spec.arrow_scale,
            arrow_minimum_mm=spec.arrow_minimum_mm,
            color_limits=color_limits,
        )
        composer = FigureComposer()
        figure, axes = composer.create(
            1,
            len(states),
            figsize=(4.2 * len(states), 4.8),
            title=spec.title,
        )
        flat_axes = list(axes.ravel())
        mesh = dataset.mesh(design_id, spec.mesh_id)
        chains = {
            side: dataset.chain(design_id, spec.mesh_id, side)
            for side in ("right", "left")
        }
        panel_metadata: list[Mapping[str, Any]] = []
        vector_rows = []
        image = plt.cm.ScalarMappable(norm=signed_norm, cmap=theme.signed_colormap)
        for axis, state in zip(flat_axes, states):
            if spec.panels.get("undeformed_mesh", True):
                panel_metadata.append(
                    MeshPanel().render(
                        axis, mesh, theme, linewidth=0.06, alpha=0.11
                    )
                )
            if spec.panels.get("deformed_observation_boundary", True):
                panel_metadata.append(
                    ObservationBoundaryOverlay().render(
                        axis,
                        chains,
                        theme,
                        displacement_by_side=state.displacement_by_side,
                        values_by_side=state.values_by_side,
                        norm=signed_norm,
                        scale_policy=scale_policy,
                    )
                )
            coordinates = np.concatenate(
                [
                    chains[side].undeformed_coordinates
                    for side in ("right", "left")
                ],
                axis=0,
            )
            vectors = np.concatenate(
                [state.displacement_by_side[side] for side in ("right", "left")],
                axis=0,
            )
            if spec.panels.get("displacement_vectors", True):
                vector_meta, selected = DisplacementVectorPanel().render(
                    axis,
                    coordinates,
                    vectors,
                    theme,
                    scale_policy,
                    maximum_arrows=spec.maximum_arrows_per_panel,
                )
                panel_metadata.append(vector_meta)
            else:
                selected = np.asarray([], dtype=int)
            if spec.panels.get("contact_annotation", True):
                panel_metadata.append(
                    ContactInputAnnotation().render(axis, state)
                )
            reaction_text = (
                "reaction unavailable"
                if state.reaction_force_n is None
                else f"F={state.reaction_force_n:.4f} N"
            )
            axis.set_title(
                rf"$\xi={state.xi:.2f}$, $\delta={state.delta_mm:.2f}$ mm"
                + f"\n{reaction_text}"
            )
            axis.text(
                0.02,
                0.02,
                f"deformation {scale_policy.deformation_scale:g}×; "
                f"arrows {scale_policy.arrow_scale:g}×",
                transform=axis.transAxes,
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
            )
            side_labels = np.asarray(
                [
                    side
                    for side in ("right", "left")
                    for _ in range(len(chains[side].eta))
                ]
            )
            eta_values = np.concatenate(
                [chains[side].eta for side in ("right", "left")]
            )
            normal_values = np.concatenate(
                [state.values_by_side[side] for side in ("right", "left")]
            )
            selected_set = set(int(index) for index in selected)
            for index in range(len(coordinates)):
                vector_rows.append(
                    (
                        design_id,
                        spec.mesh_id,
                        state.case_id,
                        state.xi,
                        state.delta_mm,
                        side_labels[index],
                        eta_values[index],
                        coordinates[index, 0],
                        coordinates[index, 1],
                        vectors[index, 0],
                        vectors[index, 1],
                        normal_values[index],
                        int(index in selected_set),
                    )
                )
        panel_metadata.append(
            SharedColorbar().add(
                figure,
                image,
                flat_axes,
                label=r"outward $u_\mathrm{normal}$ [mm]",
            )
        )
        panel_metadata.append(PanelLabelManager().apply(flat_axes, theme))
        figure.text(
            0.5,
            -0.01,
            "Gray: reconstructed undeformed Phase 4K T3 context. Colored curves and "
            "arrows use stored sidewall displacement only; no internal field is inferred.",
            ha="center",
        )
        node_rows = tuple(
            (design_id, spec.mesh_id, node_id, point[0], point[1])
            for node_id, point in zip(mesh.node_ids, mesh.node_coordinates)
        )
        element_rows = tuple(
            (
                design_id,
                spec.mesh_id,
                element_id,
                int(connectivity[0]),
                int(connectivity[1]),
                int(connectivity[2]),
            )
            for element_id, connectivity in zip(
                mesh.element_ids, mesh.element_connectivity
            )
        )
        return RenderedFigure(
            figure=figure,
            basename=spec.basename,
            figure_kind=spec.kind,
            represented_variable="physical displacement vector u and sidewall u_normal",
            units="mm",
            normalization="none",
            design_ids=spec.designs,
            mesh_ids=(spec.mesh_id,),
            cases=tuple(state.case_id for state in states),
            xi_values=tuple(state.xi for state in states),
            indentation_values_mm=(spec.indentation_mm,),
            coordinate_convention=dataset.metadata["adapters"][0][
                "coordinate_convention"
            ],
            interpolation={
                "enabled": False,
                "reason": "friendly vector atlas requires an exact stored u_xy state",
                "xi_interpolation": False,
            },
            validity={
                "displacement": "CODTM valid and finite",
                "descriptor": "reaction may be used independently; pressure descriptors not shown",
            },
            scale_policy=scale_policy,
            color_limits=color_limits,
            panel_metadata=tuple(panel_metadata),
            source_tables=(
                SourceTable(
                    "mesh_nodes.csv",
                    ("design_id", "mesh_id", "node_id", "x_mm", "y_mm"),
                    node_rows,
                    label_columns=("design_id", "mesh_id"),
                ),
                SourceTable(
                    "mesh_elements.csv",
                    (
                        "design_id",
                        "mesh_id",
                        "element_id",
                        "node_1",
                        "node_2",
                        "node_3",
                    ),
                    element_rows,
                    label_columns=("design_id", "mesh_id"),
                ),
                SourceTable(
                    "displacement_vectors.csv",
                    (
                        "design_id",
                        "mesh_id",
                        "case_id",
                        "xi",
                        "delta_mm",
                        "side",
                        "eta",
                        "X0_x_mm",
                        "X0_y_mm",
                        "u_x_mm",
                        "u_y_mm",
                        "u_normal_mm",
                        "arrow_selected",
                    ),
                    tuple(vector_rows),
                    label_columns=("design_id", "mesh_id", "case_id", "side"),
                ),
            ),
            notes=(
                "Full-volume nodal displacement was not persisted by Phase 4K.",
                "Only actual stored observation-sidewall vectors are deformed or arrowed.",
                "The undeformed T3 context was deterministically reconstructed and count-matched without a FEM solve.",
            ),
        )


def render_figure(
    dataset: VisualizationDataset,
    spec: FigureSpec,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Render and export one figure through the public framework API."""
    theme = FigureTheme.preset(spec.theme)
    theme.apply()
    if spec.kind == "transfer_map_comparison":
        rendered = TransferMapComparisonFigure().build(dataset, spec, theme)
    elif spec.kind == "displacement_vector_atlas":
        rendered = DisplacementVectorAtlasFigure().build(dataset, spec, theme)
    else:
        raise ScientificFigureError(f"unsupported builder {spec.kind}")
    target = (
        Path(output_directory)
        if output_directory is not None
        else _resolve_repository_path(spec.output_directory, spec)
    )
    return FigureExporter().export(
        rendered,
        target,
        formats=spec.export_formats,
        dpi=spec.raster_dpi,
        serialized_spec=spec.raw_spec,
        spec_path=spec.spec_path,
        source_artifacts=dataset.source_artifacts,
        source_checksums_sha256=dataset.source_checksums_sha256,
        framework_version=FRAMEWORK_VERSION,
    )
