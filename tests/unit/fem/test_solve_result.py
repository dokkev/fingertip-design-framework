from __future__ import annotations

import numpy as np
import pytest

from fem import FEAResult
from fem import results as fem_results
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


def test_element_von_mises_extraction_uses_cauchy_stress_tensor(monkeypatch) -> None:
    class FakeElement:
        def CalculateOnIntegrationPoints(self, variable, process_info):
            assert variable == "CAUCHY_STRESS_TENSOR"
            assert process_info == "process-info"
            return [
                np.asarray(
                    [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -3.0]]
                )
            ]

    class FakeModelPart:
        Elements = {7: FakeElement()}
        ProcessInfo = "process-info"

    monkeypatch.setattr(
        fem_results,
        "import_kratos",
        lambda: (type("KM", (), {"CAUCHY_STRESS_TENSOR": "CAUCHY_STRESS_TENSOR"}), None, None, None),
    )

    values = fem_results.extract_element_von_mises_stress_mpa(FakeModelPart(), [7])

    assert values[7] == pytest.approx(np.sqrt(27.0))


def test_element_von_mises_extraction_accepts_2d_cauchy_tensor(monkeypatch) -> None:
    class FakeElement:
        def CalculateOnIntegrationPoints(self, variable, process_info):
            return [np.asarray([[3.0, 2.0], [2.0, -1.0]])]

    class FakeModelPart:
        Elements = {7: FakeElement()}
        ProcessInfo = object()

    monkeypatch.setattr(
        fem_results,
        "import_kratos",
        lambda: (
            type("KM", (), {"CAUCHY_STRESS_TENSOR": "CAUCHY_STRESS_TENSOR"}),
            None,
            None,
            None,
        ),
    )
    values = fem_results.extract_element_von_mises_stress_mpa(FakeModelPart(), [7])

    assert values[7] == pytest.approx(5.0)


def test_element_von_mises_extraction_falls_back_to_cauchy_vector(monkeypatch) -> None:
    class FakeElement:
        def CalculateOnIntegrationPoints(self, variable, process_info):
            if variable == "CAUCHY_STRESS_TENSOR":
                return []
            assert variable == "CAUCHY_STRESS_VECTOR"
            return [np.asarray([3.0, 0.0, -3.0, 0.0])]

    class FakeModelPart:
        Elements = {7: FakeElement()}
        ProcessInfo = object()

    monkeypatch.setattr(
        fem_results,
        "import_kratos",
        lambda: (
            type(
                "KM",
                (),
                {
                    "CAUCHY_STRESS_TENSOR": "CAUCHY_STRESS_TENSOR",
                    "CAUCHY_STRESS_VECTOR": "CAUCHY_STRESS_VECTOR",
                },
            ),
            None,
            None,
            None,
        ),
    )
    values = fem_results.extract_element_von_mises_stress_mpa(FakeModelPart(), [7])

    assert values[7] == pytest.approx(np.sqrt(27.0))
