from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import scripts.optimization.run_bo as run_bo


def test_user_config_contains_the_production_search_controls() -> None:
    payload = run_bo._user_config_payload(trials=4)

    assert {
        "flat_pad_height",
        "semielliptical_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
        "void_height",
    }.issubset(payload["nominal_parameters"])
    assert payload["campaign_mode"] == "production"
    assert payload["trajectory_protocol"] == run_bo.USER_PROTOCOL.to_dict()
    assert payload["search_bounds"] == [
        bound.to_dict() for bound in run_bo.USER_SEARCH_BOUNDS
    ]
    assert payload["objective"]["config"] == asdict(run_bo.USER_OBJECTIVE)
    assert "nominal baseline is evaluated separately" in payload["ax"]["trials_semantics"]
    assert payload["trajectory_protocol"]["contact_locations_u"] == [
        0.25,
        0.5,
        0.75,
    ]
    assert payload["ax"]["initialization_trials"] == run_bo.INITIALIZATION_TRIALS
    assert payload["ax"]["search_trials"] == 4 - run_bo.INITIALIZATION_TRIALS
    json.dumps(payload, allow_nan=False)


def test_preflight_only_reports_external_failure_without_starting_campaign(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    requested_output = tmp_path / "custom-output"
    seen_outputs: list[object] = []

    monkeypatch.setattr(
        run_bo,
        "_preflight_payload",
        lambda output, **_kwargs: seen_outputs.append(output)
        or {
            "schema": "test",
            "status": "FAIL_EXTERNAL_PREREQUISITE",
            "failed_checks": ["optix_smoke"],
            "checks": {},
        },
    )

    assert run_bo.main(["--preflight", "--output", str(requested_output)]) == 2
    assert seen_outputs == [requested_output]
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FAIL_EXTERNAL_PREREQUISITE"


def test_preflight_success_is_a_read_only_exit(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_bo,
        "_preflight_payload",
        lambda _output, **_kwargs: {
            "schema": "test",
            "status": "PASS",
            "failed_checks": [],
            "checks": {},
        },
    )

    assert run_bo.main(["--preflight"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"


def test_preflight_builds_configuration_under_requested_output_root(
    monkeypatch,
    tmp_path,
) -> None:
    requested_output = tmp_path / "custom-output"
    requested_output.mkdir()
    marker = requested_output / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    evaluator_roots: list[object] = []

    monkeypatch.setattr(
        run_bo.importlib,
        "import_module",
        lambda _name: object(),
    )

    class _Evaluator:
        def __init__(self, root, **_kwargs):
            evaluator_roots.append(root)

    monkeypatch.setattr(run_bo, "Lumo3DTrajectoryEvaluator", _Evaluator)
    monkeypatch.setattr(
        run_bo,
        "_run_optix_smoke",
        lambda: SimpleNamespace(to_dict=lambda: {"hit": True, "miss": True}),
    )
    monkeypatch.setattr(
        run_bo,
        "_run_newton_smoke",
        lambda: {"finite_result": True},
    )

    payload = run_bo._preflight_payload(requested_output)

    assert payload["status"] == "PASS"
    assert evaluator_roots == [requested_output / "preflight-artifacts"]
    assert requested_output != run_bo.DEFAULT_OUTPUT
    assert marker.read_text(encoding="utf-8") == "keep"
