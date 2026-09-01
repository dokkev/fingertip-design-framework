"""Concrete hardware interfaces used by LUMO applications."""

from .realsense import ColorFrame, RealSenseColorCamera

__all__ = ["ColorFrame", "RealSenseColorCamera"]
