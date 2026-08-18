"""Staged characterization of the production 2D explicit-contact FEA path.

This module deliberately owns no mechanics.  It calls :func:`fem.solve.solve`,
preserves the solver's existing diagnostics, and stops the staged experiment at
the first unresolved failure boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from fem.solve import FEAResult, solve
from mesh import mesh_settings_for_level
from mesh.indenter import IndenterSettings
from model import Fingertip
from validation.common.io import atomic_write_json, strict_read_json


DEFAULT_OUTPUT = Path("output/validation/fem/contact_failure_map/summary.json")
DEFAULT_MESH_LEVEL = "medium"
DEFAULT_STEPS = 48
DEFAULT_INDENTATION_MM = 0.5
DEFAULT_RADIUS_MM = 4.0
DEFAULT_INITIAL_GAP_MM = 0.0
ISOLATION_CONTACTS = (
    "none",
    "bottom_only",
    "sides_separate",
    "three_pairs",
    "continuous_u",
)


def _basal_interface_for_contact(configuration: str) -> str:
    """Keep diagnostic contact selection separate from the basal condition."""
    if configuration in ("bottom_only", "three_pairs"):
        return "explicit_contact"
    if configuration == "sides_separate":
        return "bonded"
    return "free"
DIAGNOSTIC_API_AVAILABILITY = {
    "load_fraction": "derived from prescribed_indenter_travel_mm / requested indentation",
    "applied_pressure_mpa": "not_applicable: displacement-controlled experiment",
    "resultant_reaction_n": "available as indenter_normal_reaction_n",
    "newton_iterations": "available per converged step and failed iterate",
    "residual_norm_history": "unavailable: current Kratos strategy does not expose a stable iteration observer",
    "displacement_increment_norm": "unavailable: current production result contract does not expose it",
    "linear_solver_time_seconds": "unavailable: current production result contract does not expose it",
    "contact_group_state": "available from Kratos indexed ContactSubN/ComputingContactSubN",
    "det_f_and_strain": "available on converged pad states",
}


@dataclass(frozen=True)
class CaseSpec:
    """One production-baseline case or one first-failure isolation case."""

    stage: str
    location_x_mm: float
    indentation_mm: float
    internal_contact: str = "three_pairs"
    basal_interface: str | None = None
    steps: int = DEFAULT_STEPS
    origin_stage: str | None = None

    def __post_init__(self) -> None:
        if self.basal_interface is None:
            object.__setattr__(
                self,
                "basal_interface",
                _basal_interface_for_contact(self.internal_contact),
            )

    def key(self) -> tuple[Any, ...]:
        return (
            self.location_x_mm,
            self.indentation_mm,
            self.internal_contact,
            self.basal_interface,
            self.steps,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _baseline(stage: str, location_x_mm: float, indentation_mm: float) -> CaseSpec:
    return CaseSpec(stage, location_x_mm, indentation_mm)


def _case_record(records: Iterable[Mapping[str, Any]], spec: CaseSpec) -> Mapping[str, Any] | None:
    for record in records:
        if spec.stage == "isolation" and record.get("stage") != "isolation":
            continue
        if (
            float(record.get("x_mm", math.nan)) == spec.location_x_mm
            and float(record.get("indentation_mm", math.nan)) == spec.indentation_mm
            and record.get("internal_contact") == spec.internal_contact
            and int(record.get("steps", -1)) == spec.steps
        ):
            return record
    return None


def _status(record: Mapping[str, Any] | None) -> str | None:
    if record is None:
        return None
    return str(record.get("status", "FAIL"))


def _is_hard_stop(record: Mapping[str, Any]) -> bool:
    return bool(record.get("hard_stop", False)) or record.get("failure_category") in {
        "SETUP_OR_INITIALIZATION",
        "CONTACT_CONTRACT",
        "UNREADABLE_NONFINITE_STATE",
    }


def _missing_isolation_cases(
    records: Sequence[Mapping[str, Any]],
    failed: Mapping[str, Any],
    origin_stage: str,
) -> list[CaseSpec]:
    common = {
        "location_x_mm": float(failed["x_mm"]),
        "indentation_mm": float(failed["indentation_mm"]),
        "steps": int(failed["steps"]),
        "origin_stage": origin_stage,
    }
    return [
        CaseSpec("isolation", internal_contact=configuration, **common)
        for configuration in ISOLATION_CONTACTS
        if _case_record(
            records,
            CaseSpec(
                "isolation",
                internal_contact=configuration,
                **common,
            ),
        )
        is None
    ]


def _first_failed(records: Sequence[Mapping[str, Any]], specs: Sequence[CaseSpec]) -> Mapping[str, Any] | None:
    for spec in specs:
        record = _case_record(records, spec)
        if record is not None and _status(record) != "PASS":
            return record
    return None


def next_case_specs(records: Sequence[Mapping[str, Any]]) -> list[CaseSpec]:
    """Return only the next staged cases; never construct a Cartesian sweep."""
    nominal_center = _baseline("A", 0.0, DEFAULT_INDENTATION_MM)
    center_record = _case_record(records, nominal_center)
    if center_record is None:
        return [nominal_center]
    if _status(center_record) != "PASS":
        if _is_hard_stop(center_record):
            return []
        return _missing_isolation_cases(records, center_record, "A")

    stage_b = (
        _baseline("B", -3.0, DEFAULT_INDENTATION_MM),
        _baseline("B", 3.0, DEFAULT_INDENTATION_MM),
    )
    missing = [spec for spec in stage_b if _case_record(records, spec) is None]
    if missing:
        return [missing[0]]
    failed = _first_failed(records, stage_b)
    if failed is not None:
        if _is_hard_stop(failed):
            return []
        return _missing_isolation_cases(records, failed, "B")

    stage_c = (
        _baseline("C", 0.0, 1.0),
        _baseline("C", -3.0, 1.0),
        _baseline("C", 3.0, 1.0),
    )
    missing = [spec for spec in stage_c if _case_record(records, spec) is None]
    if missing:
        return [missing[0]]
    failed = _first_failed(records, stage_c)
    if failed is not None:
        if _is_hard_stop(failed):
            return []
        return _missing_isolation_cases(records, failed, "C")

    stage_d = (
        _baseline("D", 0.0, 1.5),
        _baseline("D", -3.0, 1.5),
        _baseline("D", 3.0, 1.5),
    )
    missing = [spec for spec in stage_d if _case_record(records, spec) is None]
    if missing:
        return [missing[0]]
    failed = _first_failed(records, stage_d)
    if failed is not None:
        if _is_hard_stop(failed):
            return []
        return _missing_isolation_cases(records, failed, "D")
    return []


def _json_safe(value: Any) -> Any:
    """Convert solver values to strict JSON without emitting NaN/Infinity."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _contact_patch_summary(result: FEAResult) -> dict[str, Any]:
    pose = result.indenter_pose
    if pose is None:
        return {
            "active_contact_node_count": None,
            "contact_patch_is_none": None,
            "contact_patch_length_mm": None,
            "CONTACT_PATCH_STATUS": "UNAVAILABLE",
        }
    patch = pose.contact_patch
    length = None if patch is None else float(patch.length)
    sufficient = patch is not None and math.isfinite(length) and length > 0.0
    return {
        "active_contact_node_count": len(pose.active_contact_node_ids),
        "contact_patch_is_none": patch is None,
        "contact_patch_length_mm": length,
        "CONTACT_PATCH_STATUS": "PASS" if sufficient else "INSUFFICIENT",
    }


def _deformation_summary(point: Mapping[str, Any] | None) -> dict[str, Any]:
    if point is None:
        return {"available": False}
    contact_groups = point.get("contact_groups", {})
    return {
        "available": True,
        "reaction_force_n": point.get("indenter_normal_reaction_n"),
        "minimum_clearance_mm_by_group": {
            name: group.get("signed_geometric_gap", {}).get("min_signed_gap_mm")
            for name, group in contact_groups.items()
        },
        "maximum_penetration_mm_by_group": {
            name: group.get("signed_geometric_gap", {}).get("maximum_penetration_mm")
            for name, group in contact_groups.items()
        },
        "maximum_pad_displacement_mm": point.get("maximum_pad_displacement_mm"),
        "contact_width": point.get("external_contact_width"),
        "pad_strain_det_f": point.get("pad_strain_det_f"),
        "volumetric_strain": point.get("volumetric_strain"),
        "rigid_indenter_validation": point.get("rigid_indenter_validation"),
    }


def _diagnostic_history(
    history: Sequence[Mapping[str, Any]],
    indentation_mm: float,
) -> list[dict[str, Any]]:
    """Annotate existing per-step records without changing solver semantics."""
    return [
        {
            **dict(point),
            "load_fraction": (
                float(point["prescribed_indenter_travel_mm"]) / indentation_mm
            ),
            "applied_pressure_mpa": None,
            "resultant_reaction_n": point.get("indenter_normal_reaction_n"),
            "residual_norm_history": None,
            "displacement_increment_norm": None,
            "linear_solver_time_seconds": None,
            "minimum_clearance_mm_by_group": {
                name: group.get("signed_geometric_gap", {}).get("min_signed_gap_mm")
                for name, group in point.get("contact_groups", {}).items()
            },
            "maximum_penetration_mm_by_group": {
                name: group.get("signed_geometric_gap", {}).get("maximum_penetration_mm")
                for name, group in point.get("contact_groups", {}).items()
            },
        }
        for point in history
    ]


def _first_penetration_violation(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for point in history:
        for name, group in point.get("contact_groups", {}).items():
            gap = group.get("signed_geometric_gap", {})
            penetration = gap.get("maximum_penetration_mm")
            tolerance = group.get("penetration_tolerance_mm")
            if (
                penetration is not None
                and tolerance is not None
                and float(penetration) > float(tolerance)
            ):
                return {
                    "group": name,
                    "step": point.get("step"),
                    "prescribed_indenter_travel_mm": point.get(
                        "prescribed_indenter_travel_mm"
                    ),
                    "maximum_penetration_mm": penetration,
                    "penetration_tolerance_mm": tolerance,
                }
    return None


def _failure_category(details: Mapping[str, Any]) -> tuple[str | None, bool]:
    if details.get("exception"):
        text = str(details["exception"]).lower()
        if "contact" in text and ("contract" in text or "modelpart" in text):
            return "CONTACT_CONTRACT", True
        return "SETUP_OR_INITIALIZATION", True
    if details.get("failure_reason") in {"non_finite_field", "nonpositive_pad_det_f"}:
        if details.get("failure_step_diagnostics", {}).get("contact_groups") is None:
            return "UNREADABLE_NONFINITE_STATE", True
    return None, False


def _make_case_record(spec: CaseSpec) -> dict[str, Any]:
    """Run one case and reduce the existing FEA diagnostics into one record."""
    started = time.perf_counter()
    indenter = IndenterSettings(
        radius_mm=DEFAULT_RADIUS_MM,
        initial_gap_mm=DEFAULT_INITIAL_GAP_MM,
    )
    try:
        tip = Fingertip()
        mesh = tip.mesh(mesh_settings_for_level(DEFAULT_MESH_LEVEL))
        result = solve(
            tip,
            mesh,
            indentation=spec.indentation_mm,
            surface_x_mm=spec.location_x_mm,
            steps=spec.steps,
            indenter=indenter,
            internal_contact=spec.internal_contact,
            basal_interface=spec.basal_interface or "free",
        )
        details = result.details
        history = _diagnostic_history(
            list(details.get("history", [])), spec.indentation_mm
        )
        last = history[-1] if history else None
        failed_iterate = details.get("failure_step_diagnostics")
        failure_category, hard_stop = _failure_category(details)
        solver_convergence_status = "PASS" if result.converged else "FAIL"
        acceptance_status = details.get("status")
        status = (
            "PASS"
            if solver_convergence_status == "PASS" and acceptance_status == "PASS"
            else "FAIL"
        )
        first_failed_step = details.get("failure_step")
        first_failed_travel = None
        if failed_iterate is not None:
            first_failed_travel = failed_iterate.get("prescribed_indenter_travel_mm")
        elif first_failed_step is not None:
            first_failed_travel = (
                spec.indentation_mm * int(first_failed_step) / spec.steps
            )
        record = {
            "stage": spec.stage,
            "origin_stage": spec.origin_stage,
            "x_mm": spec.location_x_mm,
            "indentation_mm": spec.indentation_mm,
            "internal_contact": spec.internal_contact,
            "steps": spec.steps,
            "status": status,
            "FEA_STATUS": status,
            "solver_convergence_status": solver_convergence_status,
            "solve_status": details.get("solve_status"),
            "acceptance_status": acceptance_status,
            "last_converged_step": last.get("step") if last else None,
            "last_converged_travel_mm": (
                last.get("prescribed_indenter_travel_mm") if last else None
            ),
            "first_failed_step": first_failed_step,
            "first_failed_travel_mm": first_failed_travel,
            "first_penetration_violation": _first_penetration_violation(history),
            "failure_reason": details.get("failure_reason"),
            "failure_category": failure_category,
            "hard_stop": hard_stop,
            "last_converged_state": {
                "step": last.get("step") if last else None,
                "prescribed_travel_mm": (
                    last.get("prescribed_indenter_travel_mm") if last else None
                ),
                "contact_groups": last.get("contact_groups", {}) if last else {},
                "deformation": _deformation_summary(last),
            },
            "failed_iterate": {
                "available": failed_iterate is not None,
                **({} if failed_iterate is None else _json_safe(failed_iterate)),
            },
            "contact_patch": _contact_patch_summary(result),
            "history": _json_safe(history),
            "diagnostic_api_availability": DIAGNOSTIC_API_AVAILABILITY,
            "timing": _json_safe(details.get("timing", {})),
            "case_wall_clock_seconds": time.perf_counter() - started,
            "mesh": _json_safe(details.get("mesh", {})),
            "configuration": _json_safe(details.get("configuration", {})),
            "exception": details.get("exception"),
        }
        return _json_safe(record)
    except Exception as exception:
        return {
            "stage": spec.stage,
            "origin_stage": spec.origin_stage,
            "x_mm": spec.location_x_mm,
            "indentation_mm": spec.indentation_mm,
            "internal_contact": spec.internal_contact,
            "steps": spec.steps,
            "status": "FAIL",
            "FEA_STATUS": "FAIL",
            "solver_convergence_status": "FAIL",
            "solve_status": None,
            "acceptance_status": None,
            "last_converged_step": None,
            "last_converged_travel_mm": None,
            "first_failed_step": None,
            "first_failed_travel_mm": None,
            "first_penetration_violation": None,
            "failure_reason": "exception",
            "failure_category": "SETUP_OR_INITIALIZATION",
            "hard_stop": True,
            "last_converged_state": {"available": False},
            "failed_iterate": {"available": False},
            "contact_patch": {
                "CONTACT_PATCH_STATUS": "UNAVAILABLE",
            },
            "history": [],
            "diagnostic_api_availability": DIAGNOSTIC_API_AVAILABILITY,
            "timing": {},
            "case_wall_clock_seconds": time.perf_counter() - started,
            "exception": f"{type(exception).__name__}: {exception}",
        }


def _diagnosis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = next((record for record in records if record.get("status") != "PASS"), None)
    if failed is None:
        patch_shortfall = any(
            record.get("FEA_STATUS") == "PASS"
            and record.get("contact_patch", {}).get("CONTACT_PATCH_STATUS") == "INSUFFICIENT"
            for record in records
        )
        return {
            "classification": (
                "CONTACT_PATCH_EXTRACTION_ONLY"
                if patch_shortfall
                else "NO_FAILURE_IN_TESTED_RANGE"
            ),
            "evidence": [],
        }
    if failed.get("hard_stop"):
        return {
            "classification": "NUMERICAL_FAILURE_UNRESOLVED",
            "evidence": ["diagnostic hard stop prevented topology isolation"],
        }
    variants = {
        str(record.get("internal_contact")): record
        for record in records
        if record.get("stage") == "isolation"
        and record.get("x_mm") == failed.get("x_mm")
        and record.get("indentation_mm") == failed.get("indentation_mm")
    }
    if variants.get("none", {}).get("status") == "PASS":
        if variants.get("bottom_only", {}).get("status") != "PASS":
            return {
                "classification": "INTERNAL_BOTTOM_CONTACT_ASSOCIATED",
                "evidence": [
                    f"{configuration}={variants[configuration].get('status')}"
                    for configuration in ISOLATION_CONTACTS
                    if configuration in variants
                ],
            }
        if variants.get("sides_separate", {}).get("status") != "PASS":
            return {
                "classification": "INTERNAL_SIDE_CONTACT_ASSOCIATED",
                "evidence": [
                    f"{configuration}={variants[configuration].get('status')}"
                    for configuration in ISOLATION_CONTACTS
                    if configuration in variants
                ],
            }
        if (
            variants.get("three_pairs", {}).get("status") != "PASS"
            and variants.get("continuous_u", {}).get("status") == "PASS"
        ):
            return {
                "classification": "THREE_PAIR_SEGMENTATION_ASSOCIATED",
                "evidence": [
                    f"{configuration}={variants[configuration].get('status')}"
                    for configuration in ISOLATION_CONTACTS
                    if configuration in variants
                ],
            }
    if variants.get("none", {}).get("status") != "PASS":
        return {
            "classification": "CONTACT_TOPOLOGY_NOT_PRIMARY",
            "evidence": ["none also failed at the same physical condition"],
        }
    if failed.get("failure_reason") in {"nonpositive_pad_det_f", "non_finite_field"}:
        return {
            "classification": "LARGE_DEFORMATION_ASSOCIATED",
            "evidence": ["existing deformation-validity diagnostics failed"],
        }
    return {
        "classification": "NUMERICAL_FAILURE_UNRESOLVED",
        "evidence": ["isolation results do not yet distinguish a mechanism"],
    }


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _json_safe(
        {
            "schema": 1,
            "baseline": {
                "mesh_level": DEFAULT_MESH_LEVEL,
                "mesh_settings": asdict(mesh_settings_for_level(DEFAULT_MESH_LEVEL)),
                "steps": DEFAULT_STEPS,
                "indentation_mm": DEFAULT_INDENTATION_MM,
                "indenter_radius_mm": DEFAULT_RADIUS_MM,
                "initial_gap_mm": DEFAULT_INITIAL_GAP_MM,
                "solver_settings": "current production defaults",
                "morphology": "Fingertip()",
            },
            "cases": list(records),
            "first_failure": next(
                (record for record in records if record.get("status") != "PASS"),
                None,
            ),
            "diagnosis": _diagnosis(records),
        }
    )


def _reclassify_existing_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade records written before solver/acceptance status was separated."""
    updated = dict(record)
    solver_pass = record.get(
        "solver_convergence_status",
        "PASS" if record.get("solve_status") == "PASS" else "FAIL",
    ) == "PASS"
    acceptance_pass = record.get("acceptance_status") == "PASS"
    status = "PASS" if solver_pass and acceptance_pass else "FAIL"
    updated["solver_convergence_status"] = "PASS" if solver_pass else "FAIL"
    updated["status"] = status
    updated["FEA_STATUS"] = status
    updated.setdefault("diagnostic_api_availability", DIAGNOSTIC_API_AVAILABILITY)
    if record.get("history"):
        updated["history"] = _diagnostic_history(
            record["history"], float(record["indentation_mm"])
        )
        updated["first_penetration_violation"] = _first_penetration_violation(
            updated["history"]
        )
    return updated


def run_stage(stage: str, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run one requested stage, persisting after each child case."""
    if stage not in {"A", "B", "C", "D"}:
        raise ValueError("stage must be one of A, B, C, or D")
    if output.exists():
        existing = strict_read_json(output)
        records = [_reclassify_existing_record(record) for record in existing.get("cases", [])]
    else:
        records = []
    while True:
        plans = next_case_specs(records)
        if not plans:
            break
        plan = plans[0]
        if plan.stage != stage and not (
            plan.stage == "isolation" and plan.origin_stage == stage
        ):
            break
        print(json.dumps(plan.to_dict(), sort_keys=True), flush=True)
        records.append(_make_case_record(plan))
        atomic_write_json(output, _summary(records))
        if records[-1].get("status") != "PASS" and records[-1].get("hard_stop"):
            break
    artifact = _summary(records)
    atomic_write_json(output, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = run_stage(args.stage, args.output)
    print(
        json.dumps(
            {
                "cases": len(artifact["cases"]),
                "diagnosis": artifact["diagnosis"],
                "path": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CaseSpec",
    "DEFAULT_STEPS",
    "ISOLATION_CONTACTS",
    "next_case_specs",
    "run_stage",
]
