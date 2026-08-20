from __future__ import annotations

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
    assert payload["trajectory_protocol"]["contact_locations_u"] == [0.25, 0.75]
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
        lambda output: seen_outputs.append(output)
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
        lambda _output: {
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
    study_roots: list[object] = []

    monkeypatch.setattr(
        run_bo.importlib,
        "import_module",
        lambda _name: object(),
    )

    def create_study(root, **_kwargs):
        study_roots.append(root)
        return SimpleNamespace(
            design_space=SimpleNamespace(
                active_variables=[
                    SimpleNamespace(name=SimpleNamespace(value="flat_pad_height"))
                ]
            )
        )

    monkeypatch.setattr(run_bo, "create_lumo3d_trajectory_study", create_study)
    monkeypatch.setattr(
        run_bo,
        "_run_optix_smoke",
        lambda: SimpleNamespace(to_dict=lambda: {"hit": True, "miss": True}),
    )

    payload = run_bo._preflight_payload(requested_output)

    assert payload["status"] == "PASS"
    assert study_roots == [requested_output / "preflight-artifacts"]
    assert requested_output != run_bo.DEFAULT_OUTPUT
    assert marker.read_text(encoding="utf-8") == "keep"
