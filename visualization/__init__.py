"""Geometry-aware scientific figure framework for LIT Hand artifacts."""

from visualization.data import (
    ContactCase,
    DisplacementField,
    MeshData,
    ObservationChain,
    TransferSignature,
    VisualizationDataset,
)
from visualization.adapters.phase4k_dataset import (
    load_phase4k_visualization_dataset,
)
from visualization.framework import (
    FigureSpec,
    load_figure_spec,
    load_visualization_dataset,
    render_figure,
)
from visualization.camera import plot_camera
from visualization.geometry import plot_fingertip
from visualization.transport import plot_transport

__all__ = [
    "ContactCase",
    "DisplacementField",
    "FigureSpec",
    "MeshData",
    "ObservationChain",
    "TransferSignature",
    "VisualizationDataset",
    "load_figure_spec",
    "load_phase4k_visualization_dataset",
    "load_visualization_dataset",
    "plot_camera",
    "plot_fingertip",
    "plot_transport",
    "render_figure",
]
