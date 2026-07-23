"""Sparse tangent-system diagnostics without dense rank computation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu


def analyze_sparse_system(
    matrix: sparse.spmatrix,
    rhs: Sequence[float],
    equation_map: Mapping[int, Mapping[str, Any]],
    near_zero_row_tolerance: float = 1.0e-14,
) -> dict[str, Any]:
    """Summarize CSR structure and map weak rows back to nodal DOFs."""
    if (
        not math.isfinite(near_zero_row_tolerance)
        or near_zero_row_tolerance <= 0.0
    ):
        raise ValueError("near_zero_row_tolerance must be finite and positive")
    csr = sparse.csr_matrix(matrix)
    rhs_values = np.asarray(rhs, dtype=float).reshape(-1)
    if csr.shape[0] != csr.shape[1]:
        raise ValueError("diagnostic matrix must be square")
    if rhs_values.size != csr.shape[0]:
        raise ValueError("RHS size must match the matrix")
    row_norms = np.sqrt(
        np.asarray(csr.multiply(csr).sum(axis=1), dtype=float).reshape(-1)
    )
    diagonal = np.abs(csr.diagonal())
    exact_zero_rows = np.flatnonzero(row_norms == 0.0)
    near_zero_rows = np.flatnonzero(row_norms < near_zero_row_tolerance)
    zero_diagonal = np.flatnonzero(diagonal == 0.0)

    def records(equation_ids: np.ndarray) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for equation_id in equation_ids:
            identifier = int(equation_id)
            mapped = dict(equation_map.get(identifier, {}))
            output.append(
                {
                    "equation_id": identifier,
                    "row_norm": float(row_norms[identifier]),
                    "diagonal_abs": float(diagonal[identifier]),
                    **mapped,
                }
            )
        return output

    graph = csr.copy()
    if graph.nnz:
        graph.data = np.ones(graph.nnz, dtype=np.int8)
    component_count, labels = csgraph.connected_components(
        graph, directed=False, connection="weak"
    )
    component_sizes = np.bincount(labels, minlength=component_count)
    factorization = {
        "attempted": True,
        "succeeded": False,
        "method": "scipy.sparse.linalg.splu diagnostic only",
        "exception": None,
    }
    try:
        splu(csr.tocsc())
        factorization["succeeded"] = True
    except Exception as exception:  # SuperLU reports singularity as RuntimeError.
        factorization["exception"] = (
            f"{type(exception).__name__}: {exception}"
        )
    finite_matrix = bool(np.isfinite(csr.data).all())
    finite_rhs = bool(np.isfinite(rhs_values).all())
    return {
        "shape": [int(csr.shape[0]), int(csr.shape[1])],
        "nnz": int(csr.nnz),
        "matrix_finite": finite_matrix,
        "rhs_finite": finite_rhs,
        "rhs_abs_max": (
            float(np.max(np.abs(rhs_values))) if rhs_values.size else 0.0
        ),
        "diagonal_abs_min": (
            float(np.min(diagonal)) if diagonal.size else None
        ),
        "diagonal_abs_max": (
            float(np.max(diagonal)) if diagonal.size else None
        ),
        "exact_zero_diagonal_count": int(zero_diagonal.size),
        "exact_zero_row_count": int(exact_zero_rows.size),
        "near_zero_row_count": int(near_zero_rows.size),
        "near_zero_row_tolerance": near_zero_row_tolerance,
        "exact_zero_rows": records(exact_zero_rows),
        "near_zero_rows": records(near_zero_rows),
        "zero_diagonal_rows": records(zero_diagonal),
        "equation_graph_component_count": int(component_count),
        "equation_graph_component_sizes": sorted(
            (int(value) for value in component_sizes), reverse=True
        ),
        "sparse_factorization": factorization,
    }
