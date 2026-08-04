"""Focused contracts for persisted full-field visualization artifacts."""

from __future__ import annotations

import pytest

from visualization.adapters.normal_indentation import (
    load_normal_indentation_visualization_dataset,
)
from visualization.data import ScientificFigureError


def test_missing_manifest_reports_persisted_dataset_producer(tmp_path) -> None:
    manifest = (tmp_path / "dataset_manifest.json").resolve()

    with pytest.raises(ScientificFigureError) as raised:
        load_normal_indentation_visualization_dataset(tmp_path)

    message = str(raised.value)
    assert str(manifest) in message
    assert "dataset_manifest.json" in message
    assert "normal_field_atlas" in message
