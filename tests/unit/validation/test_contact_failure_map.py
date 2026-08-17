from __future__ import annotations

import json
from validation.fem.contact_failure_map import (
    CaseSpec,
    DEFAULT_STEPS,
    ISOLATION_CONTACTS,
    _contact_patch_summary,
    _json_safe,
    _reclassify_existing_record,
    next_case_specs,
)


def _record(spec: CaseSpec, status: str = "PASS", **extra: object) -> dict[str, object]:
    value: dict[str, object] = {
        "stage": spec.stage,
        "origin_stage": spec.origin_stage,
        "x_mm": spec.location_x_mm,
        "indentation_mm": spec.indentation_mm,
        "internal_contact": spec.internal_contact,
        "steps": spec.steps,
        "status": status,
        "hard_stop": False,
    }
    value.update(extra)
    return value


def test_staging_does_not_create_cartesian_sweep() -> None:
    first = next_case_specs([])
    assert first == [CaseSpec("A", 0.0, 0.5)]

    after_a = next_case_specs([_record(first[0])])
    assert len(after_a) == 1
    assert after_a[0].stage == "B"
    assert after_a[0].location_x_mm == -3.0


def test_first_failure_triggers_only_missing_contact_isolation_cases() -> None:
    baseline = CaseSpec("A", 0.0, 0.5)
    records = [_record(baseline, "FAIL", failure_category=None)]
    plans = next_case_specs(records)

    assert [plan.internal_contact for plan in plans] == [
        "none",
        "bottom_only",
        "sides_separate",
        "three_pairs",
        "continuous_u",
    ]
    assert all(plan.stage == "isolation" for plan in plans)

    existing = records + [_record(plan, "PASS") for plan in plans[:2]]
    remaining = next_case_specs(existing)
    assert [plan.internal_contact for plan in remaining] == [
        "sides_separate",
        "three_pairs",
        "continuous_u",
    ]


def test_no_isolation_when_all_baseline_stages_pass() -> None:
    records: list[dict[str, object]] = []
    for spec in (
        CaseSpec("A", 0.0, 0.5),
        CaseSpec("B", -3.0, 0.5),
        CaseSpec("B", 3.0, 0.5),
        CaseSpec("C", 0.0, 1.0),
        CaseSpec("C", -3.0, 1.0),
        CaseSpec("C", 3.0, 1.0),
        CaseSpec("D", 0.0, 1.5),
        CaseSpec("D", -3.0, 1.5),
        CaseSpec("D", 3.0, 1.5),
    ):
        records.append(_record(spec))
    assert next_case_specs(records) == []


def test_failed_case_serialization_is_strict_json_safe() -> None:
    safe = _json_safe({"nan": float("nan"), "inf": float("inf"), "ok": 3.0})
    assert safe == {"nan": None, "inf": None, "ok": 3.0}
    json.dumps(safe, allow_nan=False)


def test_fea_pass_and_missing_contact_patch_are_separate() -> None:
    class Pose:
        active_contact_node_ids = (1, 2)
        contact_patch = None

    class Result:
        indenter_pose = Pose()

    summary = _contact_patch_summary(Result())
    assert summary["active_contact_node_count"] == 2
    assert summary["contact_patch_is_none"] is True
    assert summary["CONTACT_PATCH_STATUS"] == "INSUFFICIENT"


def test_production_step_default_remains_48() -> None:
    assert DEFAULT_STEPS == 48
    assert CaseSpec("A", 0.0, 0.5).steps == 48


def test_solver_convergence_and_production_acceptance_are_separate() -> None:
    record = _record(
        CaseSpec("D", 0.0, 1.5),
        status="PASS",
        solve_status="PASS",
        acceptance_status="FAIL",
    )
    classified = _reclassify_existing_record(record)
    assert classified["solver_convergence_status"] == "PASS"
    assert classified["FEA_STATUS"] == "FAIL"
