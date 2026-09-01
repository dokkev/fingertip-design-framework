"""Concrete hardware interfaces used by physical LUMO experiments."""

from .realsense import ColorFrame, RealSenseColorCamera

__all__ = ["ColorFrame", "RealSenseColorCamera"]
