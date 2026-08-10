from __future__ import annotations

import numpy as np
import pytest

from fem import FEAResult
from mesh import PadMesh


def _mesh() -> PadMesh:
    return PadMesh.from_arrays(
        node_ids=np.asarray([1, 2, 3], dtype=np.int64),
        reference_coordinates_mm=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        element_connectivity_node_ids=np.asarray([[1, 2, 3]], dtype=np.int64),
    )


def test_fea_result_exposes_neutral_displacement_and_deformed_mesh() -> None:
    mesh = _mesh()
    displacement = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
        dtype=float,
    )

    result = FEAResult(
        mesh=mesh,
        displacement=displacement,
        reaction_force=1.5,
        contact={"bottom": {"active": True}},
        converged=True,
        details={"solver": "synthetic"},
    )

    np.testing.assert_allclose(result.displacement, displacement)
    np.testing.assert_allclose(
        result.deformed_mesh.coordinates,
        mesh.coordinates + displacement,
    )
    assert result.deformed_mesh.reference_mesh is mesh


def test_failed_fea_result_does_not_expose_a_deformed_mesh() -> None:
    result = FEAResult(
        mesh=_mesh(),
        displacement=None,
        reaction_force=None,
        contact={},
        converged=False,
        details={"failure_reason": "not run"},
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        _ = result.deformed_mesh
