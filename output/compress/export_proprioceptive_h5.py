#!/usr/bin/env python3
"""Export or verify one upload-sized proprioceptive-force HDF5 run."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.proprioceptive_h5 import (  # noqa: E402
    export_proprioceptive_dataset_h5,
    export_proprioceptive_run_h5,
    verify_proprioceptive_dataset_h5,
    verify_proprioceptive_h5,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="source run/dataset directory or HDF5 to verify"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (default: sibling RUN_NAME.h5)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="full-resolution per-frame JPEG quality (default: 95)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=500.0,
        help="strict decimal-MB output limit (default: 500)",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="export or verify one HDF5 containing every run below a dataset root",
    )
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if not math.isfinite(args.max_size_mb) or args.max_size_mb <= 0.0:
        parser.error("--max-size-mb must be finite and positive")
    return args


def main() -> None:
    args = _arguments()
    maximum_bytes = round(args.max_size_mb * 1_000_000)
    if args.verify:
        verify = (
            verify_proprioceptive_dataset_h5
            if args.dataset
            else verify_proprioceptive_h5
        )
        summary = verify(args.run, maximum_bytes=maximum_bytes)
    else:
        output = args.output or args.run.resolve().with_suffix(".h5")
        export = (
            export_proprioceptive_dataset_h5
            if args.dataset
            else export_proprioceptive_run_h5
        )
        summary = export(
            args.run,
            output,
            jpeg_quality=args.jpeg_quality,
            maximum_bytes=maximum_bytes,
        )
    print(f"Artifact: {summary.path}")
    if args.dataset:
        print(f"Runs: {summary.run_count}")
    print(
        f"Frames: {summary.frame_count} "
        f"({summary.unloaded_reference_count} unloaded reference, "
        f"{summary.contact_frame_count} contact)"
    )
    print(f"JPEG quality: {summary.jpeg_quality}")
    print(f"Size: {summary.size_bytes / 1_000_000:.3f} MB")


if __name__ == "__main__":
    main()
