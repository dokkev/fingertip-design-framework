from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.data_collection.contact_dataset import (
    ContactDatasetWriter,
    SessionMetadata,
    SynchronizedFrame,
    iter_completed_runs,
)
from experiments.data_collection.force_sequence import ForceSequenceConfig
from experiments.hardware import BotaSample, BotaTareOffsets


def _metadata() -> SessionMetadata:
    return SessionMetadata(
        camera_model="Intel RealSense D435",
        camera_width=8,
        camera_height=6,
        camera_fps=30,
        bota_serial_port="MOCK",
        bota_tare_offsets=BotaTareOffsets(),
        force_sequence=ForceSequenceConfig(),
        sensor_mode="mock",
    )


def _frame(index: int, force_n: float = 2.0) -> SynchronizedFrame:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 17 + index
    sample = BotaSample(
        host_time_s=1.0 + index / 30.0,
        sensor_timestamp=100 + index,
        status=0,
        temperature_c=24.0,
        fx_n=0.0,
        fy_n=0.0,
        fz_n=force_n,
        mx_nm=0.0,
        my_nm=0.0,
        mz_nm=0.0,
        force_magnitude_n=force_n,
        torque_magnitude_nm=0.0,
        fz_share=1.0,
    )
    return SynchronizedFrame(
        rgb=rgb,
        camera_host_time_s=sample.host_time_s + 0.001,
        camera_device_timestamp_ms=1000.0 + index * 33.0,
        camera_frame_number=index,
        bota_sample=sample,
    )


def _run(writer: ContactDatasetWriter, hole: int = 1):
    config = ForceSequenceConfig(target_forces_n=(2.0,))
    return writer.start_loaded_run(
        morphology="DragonSkin nominal",
        indenter_type="sphere 15 mm",
        hole_index=hole,
        config=config,
        start_host_time_s=1.0,
    )


def test_completed_target_writes_png_csv_and_summary(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer)
        segment = writer.begin_force_target(run, 2.0)
        assert writer.submit_frame(segment, _frame(0))
        assert writer.submit_frame(segment, _frame(1))
        writer.finalize_segment(segment)
        writer.complete_loaded_run(run, 3.0)
        writer.flush()

        final = run.path / "force_02N"
        assert sorted(path.name for path in (final / "frames").glob("*.png")) == [
            "000000.png",
            "000001.png",
        ]
        with (final / "frames.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        summary = json.loads((final / "summary.json").read_text(encoding="utf-8"))
        assert len(rows) == 2
        assert rows[0]["rgb_filename"] == "frames/000000.png"
        assert float(rows[0]["camera_bota_time_delta_ms"]) == pytest.approx(1.0)
        assert summary["number_of_frames"] == 2
        assert summary["mean_force_magnitude_n"] == 2.0
        stored_bgr = cv2.imread(str(final / "frames" / "000000.png"))
        assert stored_bgr is not None
        assert tuple(stored_bgr[0, 0]) == (0, 0, 17)


def test_discarded_partial_is_not_a_completed_segment(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer)
        segment = writer.begin_force_target(run, 2.0)
        writer.submit_frame(segment, _frame(0))
        writer.discard_segment(segment)
        writer.abort_loaded_run(run, 2.0)
        writer.flush()

        assert not segment.partial_path.exists()
        assert not segment.final_path.exists()


def test_aborted_run_preserves_status(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer)
        writer.abort_loaded_run(run, 2.5)
        writer.flush()

        metadata = json.loads((run.path / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "aborted"
        assert metadata["run_end_host_time_s"] == 2.5


def test_reader_returns_only_completed_runs_by_default(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        completed = _run(writer, hole=1)
        first = writer.begin_force_target(completed, 2.0)
        writer.submit_frame(first, _frame(0))
        writer.finalize_segment(first)
        writer.complete_loaded_run(completed, 3.0)
        aborted = _run(writer, hole=2)
        writer.abort_loaded_run(aborted, 4.0)
        writer.flush()

    records = list(iter_completed_runs(tmp_path))
    assert [record.metadata["run_id"] for record in records] == [completed.run_id]
    assert records[0].force_segments == (completed.path / "force_02N",)


def test_hole_one_is_preserved_as_distal(tmp_path: Path) -> None:
    with ContactDatasetWriter(tmp_path, _metadata()) as writer:
        run = _run(writer, hole=1)
        writer.abort_loaded_run(run, 2.0)
        writer.flush()

        metadata = json.loads((run.path / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["hole_index"] == 1
        assert metadata["hole_index_definition"] == "1=distal, 6=proximal"
