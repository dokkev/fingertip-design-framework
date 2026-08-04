from pathlib import Path

import pytest

from validation.fingertip.indentation import normal_field_atlas


def test_atlas_sweeps_radius_at_fixed_center() -> None:
    assert normal_field_atlas.SURFACE_X_MM == 0.0
    assert normal_field_atlas.INDENTER_RADII_MM == (2.0, 4.0, 6.0)
    assert [
        normal_field_atlas._case_name(radius)
        for radius in normal_field_atlas.INDENTER_RADII_MM
    ] == ["radius_2", "radius_4", "radius_6"]
    command = normal_field_atlas._case_command(2.0, Path("/tmp/radius_2"))
    assert "--radius-mm" in command
    assert "--surface-x-mm" not in command


def test_public_api_resolves_default_output_and_returns_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, bool]] = []

    def fake_run_parent(output: Path, force: bool) -> int:
        calls.append((output, force))
        return 0

    monkeypatch.setattr(normal_field_atlas, "_run_parent", fake_run_parent)

    manifest = normal_field_atlas.run_normal_field_atlas()

    expected_output = normal_field_atlas.DEFAULT_OUTPUT.resolve()
    assert manifest == expected_output / "dataset_manifest.json"
    assert calls == [(expected_output, False)]


def test_public_api_forwards_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, bool]] = []

    def fake_run_parent(output: Path, force: bool) -> int:
        calls.append((output, force))
        return 0

    monkeypatch.setattr(normal_field_atlas, "_run_parent", fake_run_parent)

    manifest = normal_field_atlas.run_normal_field_atlas(
        tmp_path / "atlas",
        force=True,
    )

    assert manifest == (tmp_path / "atlas" / "dataset_manifest.json").resolve()
    assert calls == [((tmp_path / "atlas").resolve(), True)]


def test_public_api_raises_with_artifact_paths_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        normal_field_atlas,
        "_run_parent",
        lambda output, force: 1,
    )

    with pytest.raises(normal_field_atlas.NormalFieldAtlasError) as caught:
        normal_field_atlas.run_normal_field_atlas(tmp_path)

    message = str(caught.value)
    assert "One or more full-field FEM atlas cases failed" in message
    assert str(tmp_path.resolve()) in message
    assert str((tmp_path / "dataset_manifest.json").resolve()) in message
