"""Re-run the production LED contact detector on one recorded experiment run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.localization import LiveLedContactTracker  # noqa: E402
from experiments.analysis.proprioceptive_h5 import FORMAT_NAME  # noqa: E402


OUTPUT_COLUMNS = (
    "frame_index",
    "timestamp_ns",
    "filename",
    "ft_contact_force_N",
    "ft_timestamp_ns",
    "contact_detected",
    "contact_location_mm",
    "contact_score_z",
    "top_two_margin_DN",
    "response_led_1_DN",
    "response_led_2_DN",
    "response_led_3_DN",
    "response_led_4_DN",
    "response_led_5_DN",
    "detector_status",
)


def _read_camera_rows(
    source_path: Path,
    archive: h5py.File | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if archive is None:
        index_path = source_path / "camera_timestamps.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"camera index does not exist: {index_path}")
        with index_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = _h5_camera_rows(archive)
    if not rows:
        raise RuntimeError(f"camera index is empty: {source_path}")
    if "capture_kind" not in rows[0]:
        raise RuntimeError(
            "run predates offline-ready capture_kind metadata; collect it with "
            "--contact-processing offline or both"
        )
    references = [row for row in rows if row["capture_kind"] == "unloaded_reference"]
    contacts = [row for row in rows if row["capture_kind"] == "contact"]
    if not references:
        raise RuntimeError("run contains no unloaded_reference frames")
    if not contacts:
        raise RuntimeError("run contains no contact frames")
    references.sort(key=lambda row: int(row["timestamp_ns"]))
    contacts.sort(key=lambda row: int(row["timestamp_ns"]))
    return references, contacts


def _load_rgb(
    source_path: Path,
    row: dict[str, str],
    archive: h5py.File | None,
) -> np.ndarray:
    if archive is None:
        path = source_path / row["filename"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    else:
        index = int(row["frame_index"])
        offsets = archive["camera/jpeg_offsets"]
        start, stop = int(offsets[index]), int(offsets[index + 1])
        encoded = np.asarray(archive["camera/jpeg_bytes"][start:stop], dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read recorded frame {row['frame_index']}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _h5_camera_rows(archive: h5py.File) -> list[dict[str, str]]:
    if str(archive.attrs.get("format_name", "")) != FORMAT_NAME:
        raise ValueError("input is not a LUMO proprioceptive-force HDF5 run")
    camera = archive["camera"]
    names = (
        "frame_index",
        "timestamp_ns",
        "filename",
        "capture_kind",
        "camera_device_timestamp_ms",
        "camera_frame_number",
        "ft_contact_force_N",
        "ft_timestamp_ns",
        "camera_ft_time_delta_ms",
    )
    frame_count = int(archive.attrs["frame_count"])
    rows: list[dict[str, str]] = []
    for index in range(frame_count):
        row: dict[str, str] = {}
        for name in names:
            value = camera[name][index]
            row[name] = (
                bytes(value).decode("utf-8")
                if isinstance(value, (bytes, np.bytes_))
                else str(value)
            )
        rows.append(row)
    return rows


def process_run(run_path: Path, output_path: Path, *, overwrite: bool) -> tuple[int, int]:
    """Process saved contact frames without modifying the raw acquisition files."""

    run_path = run_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output_path}")
    if not run_path.is_dir() and not run_path.is_file():
        raise FileNotFoundError(f"input run does not exist: {run_path}")
    archive = h5py.File(run_path, "r") if run_path.is_file() else None
    try:
        references, contacts = _read_camera_rows(run_path, archive)

        tracker = LiveLedContactTracker(
            calibration_frame_count=len(references),
            baseline_frame_count=len(references),
        )
        result = None
        for row in references:
            result = tracker.process(_load_rgb(run_path, row, archive))
        if result is None or not result.geometry_ready:
            status = "no detector result" if result is None else result.status
            raise RuntimeError(f"offline LED geometry calibration failed: {status}")

        tracker.request_unloaded_baseline()
        for row in references:
            result = tracker.process(_load_rgb(run_path, row, archive))
        if result is None or not result.baseline_ready:
            status = "no detector result" if result is None else result.status
            raise RuntimeError(f"offline unloaded baseline failed: {status}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        detected_count = 0
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for source in contacts:
                try:
                    result = tracker.process(_load_rgb(run_path, source, archive))
                    response = result.optical_response_dn or ()
                    detected_count += int(result.contact_detected is True)
                    output: dict[str, object] = {
                        "frame_index": source["frame_index"],
                        "timestamp_ns": source["timestamp_ns"],
                        "filename": source["filename"],
                        "ft_contact_force_N": source["ft_contact_force_N"],
                        "ft_timestamp_ns": source["ft_timestamp_ns"],
                        "contact_detected": (
                            ""
                            if result.contact_detected is None
                            else int(result.contact_detected)
                        ),
                        "contact_location_mm": (
                            ""
                            if result.contact_location_mm is None
                            else result.contact_location_mm
                        ),
                        "contact_score_z": (
                            ""
                            if result.contact_score_z is None
                            else result.contact_score_z
                        ),
                        "top_two_margin_DN": (
                            ""
                            if result.top_two_margin_dn is None
                            else result.top_two_margin_dn
                        ),
                        "detector_status": result.status,
                    }
                    for index in range(5):
                        output[f"response_led_{index + 1}_DN"] = (
                            response[index] if index < len(response) else ""
                        )
                except Exception as error:
                    output = {
                        "frame_index": source["frame_index"],
                        "timestamp_ns": source["timestamp_ns"],
                        "filename": source["filename"],
                        "ft_contact_force_N": source["ft_contact_force_N"],
                        "ft_timestamp_ns": source["ft_timestamp_ns"],
                        "detector_status": (
                            f"error: {type(error).__name__}: {error}"
                        ),
                    }
                writer.writerow(output)
        temporary.replace(output_path)
        return len(contacts), detected_count
    finally:
        if archive is not None:
            archive.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="recorded run directory or HDF5")
    parser.add_argument(
        "--output",
        type=Path,
        help="output CSV (default: inside directory or beside HDF5)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or (
        args.run / "optical_offline.csv"
        if args.run.is_dir()
        else args.run.with_name(f"{args.run.stem}_optical_offline.csv")
    )
    frame_count, detected_count = process_run(
        args.run,
        output,
        overwrite=args.overwrite,
    )
    print(f"Wrote {output}")
    print(f"Contact detections: {detected_count}/{frame_count}")


if __name__ == "__main__":
    main()
