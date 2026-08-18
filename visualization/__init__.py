"""Thin Matplotlib plotting helpers for neutral repository artifacts."""

from visualization.camera import plot_camera
from visualization.case import plot_case_comparison
from visualization.geometry import plot_fingertip
from visualization.mesh import plot_displacement, plot_mesh
from visualization.transport import plot_transport

__all__ = [
    "plot_camera",
    "plot_case_comparison",
    "plot_displacement",
    "plot_fingertip",
    "plot_mesh",
    "plot_transport",
]
