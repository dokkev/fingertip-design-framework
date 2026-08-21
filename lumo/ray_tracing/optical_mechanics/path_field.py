"""Dependency-light accumulation for the native FULL_3D optical path field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathFieldAccumulator:
    """Accumulate weighted segment lengths into a bounded Cartesian grid."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray
    density_zyx: np.ndarray
    maximum_spacing_mm: float
    maximum_samples_per_segment: int
    processed_segment_count: int = 0

    def __post_init__(self) -> None:
        self.x_edges = self._edges("x_edges", self.x_edges)
        self.y_edges = self._edges("y_edges", self.y_edges)
        self.z_edges = self._edges("z_edges", self.z_edges)
        density = np.asarray(self.density_zyx, dtype=float)
        expected_shape = (
            len(self.z_edges) - 1,
            len(self.y_edges) - 1,
            len(self.x_edges) - 1,
        )
        if density.shape != expected_shape or not np.all(np.isfinite(density)):
            raise ValueError("density_zyx must be finite with shape (z, y, x)")
        if np.any(density < 0.0):
            raise ValueError("density_zyx must be non-negative")
        spacing = float(self.maximum_spacing_mm)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("maximum_spacing_mm must be finite and positive")
        cap = self.maximum_samples_per_segment
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError("maximum_samples_per_segment must be a positive integer")
        if (
            not isinstance(self.processed_segment_count, int)
            or isinstance(self.processed_segment_count, bool)
            or self.processed_segment_count < 0
        ):
            raise ValueError("processed_segment_count must be a non-negative integer")
        self.density_zyx = density
        self.maximum_spacing_mm = spacing

    @staticmethod
    def _edges(name: str, values: np.ndarray) -> np.ndarray:
        edges = np.asarray(values, dtype=float)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError(f"{name} must be a finite strictly increasing vector")
        return edges

    def accumulate(
        self,
        starts_mm: np.ndarray,
        ends_mm: np.ndarray,
        start_weights: np.ndarray,
        end_weights: np.ndarray,
    ) -> None:
        """Book one batch using deterministic midpoint samples."""

        starts = np.asarray(starts_mm, dtype=float)
        ends = np.asarray(ends_mm, dtype=float)
        starts_weight = np.asarray(start_weights, dtype=float)
        ends_weight = np.asarray(end_weights, dtype=float)
        if (
            starts.ndim != 2
            or starts.shape[1:] != (3,)
            or ends.shape != starts.shape
            or starts_weight.shape != (len(starts),)
            or ends_weight.shape != (len(starts),)
        ):
            raise ValueError("segment batches must have shapes (N, 3), (N,), and (N,)")
        if not all(
            np.all(np.isfinite(values))
            for values in (starts, ends, starts_weight, ends_weight)
        ):
            raise ValueError("segment batches must contain only finite values")
        if np.any(starts_weight < 0.0) or np.any(ends_weight < 0.0):
            raise ValueError("segment weights must be non-negative")
        if not len(starts):
            return

        displacement = ends - starts
        lengths = np.linalg.norm(displacement, axis=1)
        counts = np.maximum(
            1,
            np.ceil(lengths / self.maximum_spacing_mm).astype(np.int64),
        )
        counts = np.minimum(counts, self.maximum_samples_per_segment)
        sample_indices = np.arange(int(np.max(counts)), dtype=float)[None, :]
        fractions = (sample_indices + 0.5) / counts[:, None]
        samples = starts[:, None, :] + fractions[:, :, None] * displacement[:, None, :]
        x_indices = np.searchsorted(self.x_edges, samples[:, :, 0], side="right") - 1
        y_indices = np.searchsorted(self.y_edges, samples[:, :, 1], side="right") - 1
        z_indices = np.searchsorted(self.z_edges, samples[:, :, 2], side="right") - 1
        valid = sample_indices < counts[:, None]
        valid &= (
            (x_indices >= 0)
            & (x_indices < len(self.x_edges) - 1)
            & (y_indices >= 0)
            & (y_indices < len(self.y_edges) - 1)
            & (z_indices >= 0)
            & (z_indices < len(self.z_edges) - 1)
        )
        if np.any(valid):
            representative_weight = 0.5 * (starts_weight + ends_weight)
            represented = representative_weight * lengths / counts
            contributions = np.broadcast_to(represented[:, None], valid.shape)
            np.add.at(
                self.density_zyx,
                (z_indices[valid], y_indices[valid], x_indices[valid]),
                contributions[valid],
            )
        self.processed_segment_count += len(starts)

    def density_xyz(self) -> np.ndarray:
        """Return the accumulated field in the public ``(x, y, z)`` order."""

        if not np.all(np.isfinite(self.density_zyx)) or np.any(self.density_zyx < 0.0):
            raise ValueError("path-field density is non-finite or negative")
        return np.transpose(self.density_zyx, (2, 1, 0))


__all__ = ["PathFieldAccumulator"]
