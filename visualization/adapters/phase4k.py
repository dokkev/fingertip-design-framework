"""Metadata-driven post-processing helpers for Phase 4K CODTM artifacts.

The functions in this module are deliberately independent of Kratos.  They
read immutable Phase 4K artifacts, preserve the two observation sidewalls as
separate material chains, and provide the numerical operations used by the
Phase 4K-Viz command-line program.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from visualization.transforms import (
    CODTMVisualizationError,
    IndentationSelection,
    location_distance_matrix,
    select_indentation,
    shape_distance_matrix,
    signature_norm,
)


CANONICAL_INPUT_FILES = (
    "codtm_arrays.npz",
    "codtm_long.csv",
    "map_metadata.json",
    "summary.json",
    "validation.json",
    "case_summary.csv",
    "source_trace.json",
)




@dataclass(frozen=True)
class CaseRecord:
    """Case identity reconstructed from the canonical long-form CSV."""

    name: str
    mesh: str
    xi_cmd: float


@dataclass
class CODTMDataset:
    """Canonical in-memory view of the Phase 4K data."""

    input_dir: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    case_order: tuple[str, ...]
    side_order: tuple[str, ...]
    cases: dict[str, CaseRecord]
    reference_xy: dict[tuple[str, str], np.ndarray]

    def side_index(self, side: str) -> int:
        try:
            return self.side_order.index(side)
        except ValueError as exc:
            raise CODTMVisualizationError(f"unknown observation side {side!r}") from exc

    def case_index(self, case: str) -> int:
        try:
            return self.case_order.index(case)
        except ValueError as exc:
            raise CODTMVisualizationError(f"unknown CODTM case {case!r}") from exc

    def cases_for_mesh(self, mesh: str) -> tuple[CaseRecord, ...]:
        selected = [case for case in self.cases.values() if case.mesh == mesh]
        if not selected:
            raise CODTMVisualizationError(f"no cases found for mesh {mesh!r}")
        return tuple(sorted(selected, key=lambda item: (item.xi_cmd, item.name)))

    def eta_for_side(self, side: str) -> np.ndarray:
        eta = canonicalize_array(
            self.arrays["eta"],
            self.metadata["array_axes"]["eta"]["axes"],
            ("side", "observation_sample"),
        )
        return np.asarray(eta[self.side_index(side)], dtype=float)

    def canonical_field(self, name: str) -> np.ndarray:
        """Return a named field in its documented semantic axis order."""
        desired = {
            "u_normal": ("case", "step", "side", "observation_sample"),
            "u_tangent": ("case", "step", "side", "observation_sample"),
            "G_secant": ("case", "step", "side", "observation_sample"),
            "G_tangent": ("case", "step", "side", "observation_sample"),
            "u_xy": (
                "case",
                "step",
                "side",
                "observation_sample",
                "xy_component",
            ),
            "delta_n": ("case", "step"),
            "F_n": ("case", "step"),
        }
        if name not in desired:
            raise CODTMVisualizationError(f"no canonical axis contract for {name!r}")
        descriptor = self.metadata["array_axes"].get(name)
        if not isinstance(descriptor, Mapping) or "axes" not in descriptor:
            raise CODTMVisualizationError(f"metadata does not declare axes for {name!r}")
        return canonicalize_array(self.arrays[name], descriptor["axes"], desired[name])

    def valid_for_case(self, case: str) -> np.ndarray:
        index = self.case_index(case)
        mask = np.asarray(self.arrays["valid_mask"], dtype=bool)
        expected = self.canonical_field("delta_n").shape
        if mask.shape != expected:
            raise CODTMVisualizationError(
                f"valid_mask shape {mask.shape} does not match case/step shape {expected}"
            )
        return mask[index]

    def select_case_field(
        self,
        case: str,
        field: str,
        target_mm: float,
        *,
        tolerance: float = 1.0e-10,
    ) -> IndentationSelection:
        case_index = self.case_index(case)
        delta = self.canonical_field("delta_n")[case_index]
        values = self.canonical_field(field)[case_index]
        return select_indentation(
            delta,
            values,
            self.valid_for_case(case),
            target_mm,
            tolerance=tolerance,
        )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise CODTMVisualizationError(f"invalid strict JSON: {path}") from exc


def canonicalize_array(
    values: np.ndarray,
    actual_axes: Sequence[str],
    desired_axes: Sequence[str],
) -> np.ndarray:
    """Transpose an array using named axes rather than positional assumptions."""
    actual = tuple(str(axis) for axis in actual_axes)
    desired = tuple(str(axis) for axis in desired_axes)
    if len(actual) != values.ndim or len(set(actual)) != len(actual):
        raise CODTMVisualizationError("array axes are missing, duplicated, or rank-invalid")
    if set(actual) != set(desired):
        raise CODTMVisualizationError(
            f"array axes {actual!r} do not match requested axes {desired!r}"
        )
    return np.transpose(values, tuple(actual.index(axis) for axis in desired))


def input_checksums(input_dir: Path) -> dict[str, str]:
    """Return SHA-256 checksums for every canonical Phase 4K input."""
    result: dict[str, str] = {}
    for name in CANONICAL_INPUT_FILES:
        path = input_dir / name
        if not path.is_file():
            raise CODTMVisualizationError(f"missing canonical input {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def _case_records_from_csv(
    path: Path,
) -> tuple[dict[str, CaseRecord], dict[tuple[str, str], np.ndarray], int]:
    identities: dict[str, tuple[str, float]] = {}
    reference_rows: dict[tuple[str, str], dict[float, tuple[float, float]]] = {}
    row_count = 0
    required = {
        "case",
        "mesh",
        "step",
        "xi_cmd",
        "side_name",
        "eta",
        "X0_x",
        "X0_y",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise CODTMVisualizationError("codtm_long.csv is missing required columns")
        for row in reader:
            row_count += 1
            case = row["case"]
            identity = (row["mesh"], float(row["xi_cmd"]))
            if case in identities and identities[case] != identity:
                raise CODTMVisualizationError(f"inconsistent identity for case {case}")
            identities[case] = identity
            key = (case, row["side_name"])
            eta = float(row["eta"])
            point = (float(row["X0_x"]), float(row["X0_y"]))
            prior = reference_rows.setdefault(key, {}).get(eta)
            if prior is not None and not np.allclose(prior, point, rtol=0.0, atol=1e-12):
                raise CODTMVisualizationError(
                    f"reference coordinate changes for {case}/{row['side_name']}"
                )
            reference_rows[key][eta] = point
    cases = {
        name: CaseRecord(name=name, mesh=mesh, xi_cmd=xi)
        for name, (mesh, xi) in identities.items()
    }
    references = {
        key: np.asarray(
            [point for _, point in sorted(rows.items(), key=lambda item: item[0])],
            dtype=float,
        )
        for key, rows in reference_rows.items()
    }
    return cases, references, row_count


def load_codtm_dataset(input_dir: Path | str) -> tuple[CODTMDataset, dict[str, Any]]:
    """Load and cross-check all canonical Phase 4K artifacts without mutation."""
    root = Path(input_dir).resolve()
    checksums = input_checksums(root)
    metadata = _strict_json(root / "map_metadata.json")
    validation = _strict_json(root / "validation.json")
    if metadata.get("phase") != "4K" or validation.get("status") != "PASS":
        raise CODTMVisualizationError("Phase 4K metadata/validation is not PASS")
    axes = metadata.get("array_axes")
    if not isinstance(axes, Mapping):
        raise CODTMVisualizationError("map_metadata.json has no array_axes object")
    case_order = tuple(str(value) for value in axes.get("case_order", ()))
    side_order = tuple(str(value) for value in axes.get("side_order", ()))
    if not case_order or set(side_order) != {"left", "right"}:
        raise CODTMVisualizationError("invalid case_order or side_order")
    with np.load(root / "codtm_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required_arrays = {
        "u_normal",
        "u_tangent",
        "u_xy",
        "F_n",
        "delta_n",
        "eta",
        "xi_cmd",
        "xi_centroid",
        "contact_length",
        "valid_mask",
        "G_secant",
        "G_tangent",
    }
    if not required_arrays.issubset(arrays):
        raise CODTMVisualizationError(
            f"NPZ is missing arrays {sorted(required_arrays - arrays.keys())}"
        )
    for name, descriptor in axes.items():
        if not isinstance(descriptor, Mapping) or name not in arrays:
            continue
        declared_shape = tuple(int(value) for value in descriptor.get("shape", ()))
        if declared_shape and arrays[name].shape != declared_shape:
            raise CODTMVisualizationError(
                f"{name} shape {arrays[name].shape} differs from metadata {declared_shape}"
            )
    cases, reference_xy, csv_row_count = _case_records_from_csv(
        root / "codtm_long.csv"
    )
    if set(cases) != set(case_order):
        raise CODTMVisualizationError("CSV/metadata case sets differ")
    dataset = CODTMDataset(
        input_dir=root,
        metadata=metadata,
        arrays=arrays,
        case_order=case_order,
        side_order=side_order,
        cases=cases,
        reference_xy=reference_xy,
    )
    u_normal = dataset.canonical_field("u_normal")
    delta = dataset.canonical_field("delta_n")
    eta = canonicalize_array(
        arrays["eta"], axes["eta"]["axes"], ("side", "observation_sample")
    )
    if u_normal.shape[:2] != delta.shape or u_normal.shape[2:] != eta.shape:
        raise CODTMVisualizationError("NPZ case/step/side/sample dimensions disagree")
    expected_rows = int(np.prod(u_normal.shape))
    if csv_row_count != expected_rows:
        raise CODTMVisualizationError(
            f"CSV has {csv_row_count} rows; expected {expected_rows}"
        )
    for case in case_order:
        for side in side_order:
            reference = reference_xy.get((case, side))
            if reference is None or reference.shape != (eta.shape[1], 2):
                raise CODTMVisualizationError(
                    f"reference samples invalid for {case}/{side}"
                )
    valid = np.asarray(arrays["valid_mask"], dtype=bool)
    finite_valid = bool(np.isfinite(u_normal[valid]).all())
    audit = {
        "status": "PASS" if finite_valid else "FAIL",
        "input_directory": str(root),
        "input_checksums_sha256": checksums,
        "case_count": len(case_order),
        "step_count": int(u_normal.shape[1]),
        "side_count": int(u_normal.shape[2]),
        "samples_per_side": int(u_normal.shape[3]),
        "csv_row_count": csv_row_count,
        "npz_csv_expected_row_count": expected_rows,
        "all_valid_displacements_finite": finite_valid,
        "valid_step_count": int(valid.sum()),
        "contact_descriptor_semantics": {
            "displacement_validity": axes["valid_mask"]["meaning"],
            "xi_centroid_nan_meaning": axes["xi_centroid"]["nan_meaning"],
            "contact_length_nan_meaning": axes["contact_length"]["nan_meaning"],
        },
    }
    if not finite_valid:
        raise CODTMVisualizationError("valid CODTM displacement field is non-finite")
    return dataset, audit


def zeta_for_side(side: str, eta: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the exact visualization-only signed coordinate."""
    values = np.asarray(eta, dtype=float)
    if side == "right":
        return values - 1.0
    if side == "left":
        return 1.0 - values
    raise CODTMVisualizationError(f"unknown side {side!r}")


def display_zeta_for_side(
    side: str,
    eta: Sequence[float] | np.ndarray,
    *,
    gap_width: float = 0.08,
) -> np.ndarray:
    """Map exact zeta to two disjoint rendered segments with a visible gap."""
    if not 0.0 < gap_width < 1.0:
        raise CODTMVisualizationError("display gap must lie in (0, 1)")
    zeta = zeta_for_side(side, eta)
    half = 0.5 * gap_width
    scale = 1.0 - half
    return zeta * scale + (-half if side == "right" else half)


def profile_segments(
    values_by_side: Mapping[str, Sequence[float] | np.ndarray],
    eta_by_side: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """Return independent right/left segments; never a center-connected curve."""
    return tuple(
        (
            side,
            zeta_for_side(side, eta_by_side[side]),
            np.asarray(values_by_side[side], dtype=float),
        )
        for side in ("right", "left")
    )


def mirror_signature(
    signature: np.ndarray,
    side_order: Sequence[str],
) -> np.ndarray:
    """Swap semantic left/right sides while preserving each side's eta order."""
    values = np.asarray(signature, dtype=float)
    order = tuple(side_order)
    if values.ndim != 2 or set(order) != {"left", "right"}:
        raise CODTMVisualizationError("mirror input must contain left and right sides")
    return np.stack(
        [values[order.index("right")], values[order.index("left")]], axis=0
    )


def mirror_metrics(
    original: np.ndarray,
    partner: np.ndarray,
    eta_by_side: np.ndarray,
    side_order: Sequence[str],
    *,
    norm_floor: float = 1.0e-12,
) -> dict[str, Any]:
    """Compare a signature with the eta-preserving mirror of its partner."""
    order = tuple(side_order)
    mirrored_semantic = mirror_signature(partner, order)
    mirrored = np.empty_like(np.asarray(original, dtype=float))
    mirrored[order.index("left")] = mirrored_semantic[0]
    mirrored[order.index("right")] = mirrored_semantic[1]
    residual = np.asarray(original, dtype=float) - mirrored
    absolute = signature_norm(residual, eta_by_side)
    denominator = signature_norm(np.asarray(original, dtype=float), eta_by_side)
    return {
        "mirrored": mirrored,
        "residual": residual,
        "absolute_l2_mm": absolute,
        "relative_l2": absolute / max(denominator, norm_floor),
        "max_abs_mm": float(np.max(np.abs(residual))),
    }


def common_eta_profiles(
    source_values: np.ndarray,
    source_eta: np.ndarray,
    target_eta: np.ndarray,
) -> np.ndarray:
    """Interpolate sidewise profiles only in eta, never in contact location."""
    values = np.asarray(source_values, dtype=float)
    source = np.asarray(source_eta, dtype=float)
    target = np.asarray(target_eta, dtype=float)
    if values.shape != source.shape or source.shape[0] != target.shape[0]:
        raise CODTMVisualizationError("invalid common-eta profile shapes")
    result = np.empty_like(target, dtype=float)
    for side in range(source.shape[0]):
        sorted_values, sorted_eta = _sorted_side(values[side], source[side])
        if target[side].min() < sorted_eta.min() or target[side].max() > sorted_eta.max():
            raise CODTMVisualizationError("eta extrapolation is forbidden")
        result[side] = np.interp(target[side], sorted_eta, sorted_values)
    return result


def profile_comparison_metrics(
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    floor: float = 1.0e-12,
) -> dict[str, float]:
    first = np.asarray(reference, dtype=float).ravel()
    second = np.asarray(comparison, dtype=float).ravel()
    residual = first - second
    denominator = max(float(np.linalg.norm(second)), floor * math.sqrt(second.size))
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    correlation_denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    correlation = (
        float(np.dot(first_centered, second_centered) / correlation_denominator)
        if correlation_denominator > floor
        else float("nan")
    )
    return {
        "relative_l2": float(np.linalg.norm(residual)) / denominator,
        "max_abs_mm": float(np.max(np.abs(residual))),
        "shape_correlation": correlation,
    }


def descriptor_verified_mask(dataset: CODTMDataset) -> np.ndarray:
    """Return the finite contact-descriptor mask, separate from CODTM validity."""
    centroid = np.asarray(dataset.arrays["xi_centroid"], dtype=float)
    length = np.asarray(dataset.arrays["contact_length"], dtype=float)
    return np.isfinite(centroid) & np.isfinite(length)


def independent_tangent_gain(
    delta_mm: np.ndarray,
    u_normal: np.ndarray,
) -> np.ndarray:
    """Recompute the documented centered/one-sided finite-difference gain."""
    return np.gradient(
        np.asarray(u_normal, dtype=float),
        np.asarray(delta_mm, dtype=float),
        axis=0,
        edge_order=1,
    )


def finite_csv_audit(path: Path, nonnumeric: Iterable[str] = ()) -> dict[str, Any]:
    """Check that all non-label source-data cells are finite numbers."""
    labels = set(nonnumeric)
    rows = 0
    columns: tuple[str, ...] = ()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        for row in reader:
            rows += 1
            for key, value in row.items():
                if key in labels:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise CODTMVisualizationError(
                        f"{path.name}:{rows + 1} nonnumeric {key}={value!r}"
                    ) from exc
                if not math.isfinite(number):
                    raise CODTMVisualizationError(
                        f"{path.name}:{rows + 1} non-finite {key}"
                    )
    return {"path": str(path), "row_count": rows, "columns": list(columns), "finite": True}
