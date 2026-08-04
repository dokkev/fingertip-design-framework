from pathlib import Path

import pytest

from examples import fem_visualize


def test_main_runs_producer_then_loads_and_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    specification = object()
    dataset = object()

    monkeypatch.setattr(
        fem_visualize,
        "run_normal_field_atlas",
        lambda: events.append(("produce", None)) or Path("dataset_manifest.json"),
    )
    monkeypatch.setattr(
        fem_visualize,
        "load_figure_spec",
        lambda path: events.append(("spec", path)) or specification,
    )
    monkeypatch.setattr(
        fem_visualize,
        "load_visualization_dataset",
        lambda spec: events.append(("dataset", spec)) or dataset,
    )
    monkeypatch.setattr(
        fem_visualize,
        "show_figure",
        lambda loaded, spec: events.append(("show", (loaded, spec))),
    )

    assert fem_visualize.main() == 0
    assert events == [
        ("produce", None),
        (
            "spec",
            fem_visualize.repository_root
            / "examples"
            / "displacement_vector_atlas.yaml",
        ),
        ("dataset", specification),
        ("show", (dataset, specification)),
    ]
