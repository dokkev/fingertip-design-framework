"""Thin Matplotlib plotting helpers for neutral repository artifacts."""

from visualization.camera import plot_camera
from visualization.case import plot_case_comparison
from visualization.geometry import plot_fingertip
from visualization.mechanics import plot_fea
from visualization.mesh import plot_mesh
from visualization.optics import plot_transport

__all__ = [
    "plot_camera",
    "plot_case_comparison",
    "plot_fea",
    "plot_fingertip",
    "plot_mesh",
    "plot_transport",
]
