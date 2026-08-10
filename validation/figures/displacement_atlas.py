"""Render the persisted normal-indentation displacement atlas directly."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from optics.adapters import load_pad_mesh_npz
from validation.common.io import strict_read_json
from visualization import plot_displacement


class DisplacementAtlasError(RuntimeError):
    """Raised when persisted atlas artifacts do not satisfy their contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cases(input_directory: Path) -> list[dict[str, Any]]:
    manifest_path = input_directory / "dataset_manifest.json"
    try:
        manifest = strict_read_json(manifest_path)
    except (OSError, ValueError) as exc:
        raise DisplacementAtlasError(
            f"cannot read displacement atlas manifest: {manifest_path}"
        ) from exc
    if (
        manifest.get("phase") != "normal_indentation_full_field"
        or manifest.get("status") != "PASS"
    ):
        raise DisplacementAtlasError("displacement atlas manifest is not PASS")
    records = manifest.get("cases")
    if not isinstance(records, list) or not records:
        raise DisplacementAtlasError("displacement atlas has no cases")

    loaded: list[dict[str, Any]] = []
    reference_signature: tuple[Any, ...] | None = None
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "PASS":
            raise DisplacementAtlasError("every displacement atlas case must be PASS")
        result_path = input_directory / str(record["result"])
        field_path = input_directory / str(record["field"])
        if _sha256(result_path) != record.get("result_sha256"):
            raise DisplacementAtlasError(f"result checksum mismatch: {result_path}")
        if _sha256(field_path) != record.get("field_sha256"):
            raise DisplacementAtlasError(f"field checksum mismatch: {field_path}")
        result = strict_read_json(result_path)
        if (
            result.get("phase") != "normal_indentation_full_field"
            or result.get("status") != "PASS"
            or result.get("solve_status") != "PASS"
        ):
            raise DisplacementAtlasError(f"invalid case result: {result_path}")
        try:
            with np.load(field_path, allow_pickle=False) as source:
                displacement = np.asarray(source["displacement_mm"], dtype=float)
                stored_magnitude = np.asarray(
                    source["displacement_magnitude_mm"], dtype=float
                )
        except (OSError, KeyError, ValueError) as exc:
            raise DisplacementAtlasError(f"invalid field artifact: {field_path}") from exc
        if (
            displacement.ndim != 2
            or displacement.shape[1] != 2
            or not np.all(np.isfinite(displacement))
            or not np.allclose(stored_magnitude, np.linalg.norm(displacement, axis=1))
        ):
            raise DisplacementAtlasError(f"invalid displacement field: {field_path}")
        try:
            loaded_mesh = load_pad_mesh_npz(field_path, metadata={"source": str(field_path)})
        except ValueError as exc:
            raise DisplacementAtlasError(f"invalid pad mesh field: {field_path}") from exc
        reference = loaded_mesh.reference_mesh
        signature = (
            tuple(reference.node_ids.tolist()),
            reference.coordinates.tobytes(),
            reference.triangles.tobytes(),
            tuple(
                (tag, reference.boundaries[tag].tobytes())
                for tag in reference.semantic_boundary_tags
            ),
        )
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise DisplacementAtlasError("atlas cases do not share one pad topology")
        loaded.append(
            {
                "radius_mm": float(record["indenter_radius_mm"]),
                "result": result,
                "mesh": reference,
                "displacement": loaded_mesh.displacement,
            }
        )
    return sorted(loaded, key=lambda item: item["radius_mm"])


def render_displacement_atlas(
    input_directory: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Load persisted full-field artifacts and save one direct Matplotlib figure."""
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    cases = _load_cases(Path(input_directory).expanduser().resolve())
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    common_max = max(
        float(np.max(np.linalg.norm(case["displacement"], axis=1)))
        for case in cases
    )
    figure, axes = plt.subplots(
        1,
        len(cases),
        figsize=(4.5 * len(cases), 5.0),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, case in zip(axes.flat, cases, strict=True):
        result = case["result"]
        direction = np.asarray(
            result["configuration"]["indenter"]["loading_direction"],
            dtype=float,
        )
        contact = np.asarray(result["actual_surface_point_mm"], dtype=float)
        delta = float(result["final"]["achieved_indentation_mm"])
        plot_displacement(
            case["mesh"],
            case["displacement"],
            ax=axis,
            deformation_scale=1.0,
            arrow_scale=1.0,
            arrow_minimum_mm=0.001,
            maximum_arrows=80,
            normalization_max=common_max if common_max > 0.0 else 1.0,
            contact_point=contact + delta * direction,
            indentation_direction=direction,
            title=f"R={case['radius_mm']:g} mm, δ={delta:.2f} mm",
        )
    figure.suptitle("Centered indentation radius sweep: full FEM displacement field")
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/validation/fingertip/indentation/normal_full_field"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figures/displacement_vector_atlas/displacement_vector_atlas.png"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args()
    print(render_displacement_atlas(arguments.input_dir, arguments.output, dpi=arguments.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
