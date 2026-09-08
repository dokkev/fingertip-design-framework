from __future__ import annotations

import csv
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from experiments.analysis.proprioceptive_h5 import (
    export_proprioceptive_run_h5,
    verify_proprioceptive_h5,
)
from experiments.localization import LiveLedContactResult
from scripts import process_proprioceptive_contact_offline as offline


def _make_run(root: Path) -> Path:
    run = root / "run_001"
    camera = run / "camera"
    camera.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, kind in enumerate(
        ("unloaded_reference", "unloaded_reference", "contact")
    ):
        image = np.full((24, 32, 3), 20 + 30 * index, dtype=np.uint8)
        filename = f"camera/frame_{index:06d}.png"
        assert cv2.imwrite(str(run / filename), image)
        rows.append(
            {
                "frame_index": index,
                "timestamp_ns": 100 + index,
                "filename": filename,
                "capture_kind": kind,
                "camera_device_timestamp_ms": 10.0 + index,
                "camera_frame_number": index,
                "ft_contact_force_N": 0.0 if index < 2 else 2.0,
                "ft_timestamp_ns": 90 + index,
                "camera_ft_time_delta_ms": 0.01,
            }
        )
    with (run / "camera_timestamps.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run / "metadata.json").write_text('{"run_id":"run_001"}\n', encoding="utf-8")
    return run


def test_export_preserves_all_frames_and_stays_verifiable(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    output = tmp_path / "run_001.h5"
    summary = export_proprioceptive_run_h5(run, output)

    assert summary.frame_count == 3
    assert summary.unloaded_reference_count == 2
    assert summary.contact_frame_count == 1
    assert summary.size_bytes < 500_000_000
    assert verify_proprioceptive_h5(output) == summary
    with h5py.File(output, "r") as archive:
        references, contacts = offline._read_camera_rows(output, archive)
        assert len(references) == 2
        assert len(contacts) == 1
        decoded = offline._load_rgb(output, contacts[0], archive)
    assert decoded.shape == (24, 32, 3)


def test_export_removes_partial_when_size_limit_is_exceeded(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    output = tmp_path / "run_001.h5"
    with pytest.raises(RuntimeError, match="exceeding"):
        export_proprioceptive_run_h5(run, output, maximum_bytes=1)
    assert not output.exists()
    assert not output.with_suffix(".h5.partial").exists()


class _FakeTracker:
    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.baseline_ready = False

    def request_unloaded_baseline(self) -> None:
        self.baseline_ready = True

    def process(self, rgb: np.ndarray) -> LiveLedContactResult:
        del rgb
        return LiveLedContactResult(
            status="contact" if self.baseline_ready else "geometry ready",
            geometry_ready=True,
            baseline_ready=self.baseline_ready,
            contact_detected=True if self.baseline_ready else None,
            contact_location_mm=20.0 if self.baseline_ready else None,
            contact_score_z=6.0 if self.baseline_ready else None,
            top_two_margin_dn=2.0 if self.baseline_ready else None,
            optical_response_dn=(1.0, 2.0, 3.0, 2.0, 1.0),
            landmarks_xy_px=None,
            contact_point_xy_px=None,
        )


def test_offline_processor_accepts_hdf5_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _make_run(tmp_path)
    archive = tmp_path / "run_001.h5"
    export_proprioceptive_run_h5(run, archive)
    monkeypatch.setattr(offline, "LiveLedContactTracker", _FakeTracker)

    output = tmp_path / "offline.csv"
    frame_count, detected_count = offline.process_run(
        archive, output, overwrite=False
    )
    assert (frame_count, detected_count) == (1, 1)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["contact_location_mm"] == "20.0"
