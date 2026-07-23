"""Deterministic numerical source, raster/vector, and manifest export."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from visualization.data import ScientificFigureError
from visualization.theme import ScalePolicy


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
    surface_x_values_mm: tuple[float, ...] = ()
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
            "surface_x_values_mm": list(rendered.surface_x_values_mm),
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
