#!/usr/bin/env python3
"""Create upload-sized HDF5 copies of final physical observations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.compact_export import (  # noqa: E402
    export_compact_physical_data,
    verify_compact_hdf5,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve every final physical observation as compact canonical RGB, "
            "Green profile, force, timing, and source metadata in two HDF5 files."
        )
    )
    parser.add_argument(
        "--contact-dataset-root",
        type=Path,
        default=REPOSITORY_ROOT / "output" / "contact_dataset",
    )
    parser.add_argument(
        "--contact-history-root",
        type=Path,
        default=REPOSITORY_ROOT / "output" / "contact_history",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "output" / "upload",
    )
    parser.add_argument(
        "--verify",
        nargs="+",
        type=Path,
        help="verify existing compact HDF5 files without exporting",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.verify:
        for path in args.verify:
            result = verify_compact_hdf5(path)
            print(
                f"{path}: {result['session_count']} sessions, "
                f"{result['frame_count']} observations, "
                f"{result['calibration_count']} unloaded calibrations"
            )
        return

    contact, history = export_compact_physical_data(
        args.contact_dataset_root,
        args.contact_history_root,
        args.output,
    )
    print(f"Contact dataset: {contact} ({contact.stat().st_size / 1_000_000:.3f} MB)")
    print(f"Contact history: {history} ({history.stat().st_size / 1_000_000:.3f} MB)")
    print(
        "Combined: "
        f"{(contact.stat().st_size + history.stat().st_size) / 1_000_000:.3f} MB"
    )


if __name__ == "__main__":
    main()
