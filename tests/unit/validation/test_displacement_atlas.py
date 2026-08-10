"""Direct validation-figure artifact checks without solver dependencies."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from validation.figures.displacement_atlas import render_displacement_atlas


def test_displacement_atlas_loads_persisted_pad_artifact(tmp_path) -> None:
    case = tmp_path / "radius_2"
    case.mkdir()
    node_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    displacement = np.asarray(
        [[0.0, 0.0], [0.02, 0.0], [0.02, 0.02], [0.0, 0.02]],
        dtype=float,
    )
    field = case / "full_pad_field.npz"
    np.savez(
        field,
        node_ids=node_ids,
        reference_coordinates_mm=coordinates,
        element_ids=np.asarray([1, 2], dtype=np.int64),
        element_connectivity_node_ids=np.asarray(
            [[10, 20, 30], [10, 30, 40]], dtype=np.int64
        ),
        displacement_mm=displacement,
        displacement_magnitude_mm=np.linalg.norm(displacement, axis=1),
        boundary_edge_node_ids__bottom=np.asarray([[10, 20]], dtype=np.int64),
        boundary_edge_node_ids__right=np.asarray([[20, 30]], dtype=np.int64),
        boundary_edge_node_ids__top=np.asarray([[30, 40]], dtype=np.int64),
        boundary_edge_node_ids__left=np.asarray([[40, 10]], dtype=np.int64),
    )
    result = case / "result.json"
    result.write_text(
        json.dumps(
            {
                "phase": "normal_indentation_full_field",
                "status": "PASS",
                "solve_status": "PASS",
                "configuration": {"indenter": {"loading_direction": [0.0, 1.0]}},
                "actual_surface_point_mm": [0.5, 0.0],
                "final": {"achieved_indentation_mm": 0.1},
            }
        ),
        encoding="utf-8",
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "phase": "normal_indentation_full_field",
                "status": "PASS",
                "cases": [
                    {
                        "status": "PASS",
                        "indenter_radius_mm": 2.0,
                        "result": "radius_2/result.json",
                        "field": "radius_2/full_pad_field.npz",
                        "result_sha256": digest(result),
                        "field_sha256": digest(field),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = render_displacement_atlas(tmp_path, tmp_path / "atlas.png", dpi=100)
    assert output.is_file()
    assert output.stat().st_size > 0
