"""Composable Matplotlib helpers for LUMO publication figures."""

from .ablation import (
    plot_carrier_identity_comparison,
    plot_pareto_small_multiple,
    plot_structural_ablation_schematic,
    plot_void_coupled_response,
)
from .fingertip import plot_fingertip_parameterization
from .layout import (
    add_figure_box,
    add_panel_labels,
    create_figure,
    create_gridspec,
    render_panel,
    save_figure,
)
from .panels import (
    plot_contact_area,
    plot_force_displacement,
    plot_image,
    plot_incremental_stiffness,
    plot_optical_response,
    plot_pareto,
)
from .style import (
    DEFAULT_STYLE,
    STATUS_MARKERS,
    PublicationStyle,
    SemanticColors,
    material_color,
    publication_context,
)

__all__ = [
    "DEFAULT_STYLE",
    "STATUS_MARKERS",
    "PublicationStyle",
    "SemanticColors",
    "add_figure_box",
    "add_panel_labels",
    "create_figure",
    "create_gridspec",
    "material_color",
    "plot_carrier_identity_comparison",
    "plot_contact_area",
    "plot_force_displacement",
    "plot_fingertip_parameterization",
    "plot_image",
    "plot_incremental_stiffness",
    "plot_optical_response",
    "plot_pareto",
    "plot_pareto_small_multiple",
    "plot_structural_ablation_schematic",
    "plot_void_coupled_response",
    "publication_context",
    "render_panel",
    "save_figure",
]
