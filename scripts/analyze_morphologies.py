#!/usr/bin/env python3
"""Analyze one or more LUMO physical morphology sessions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.summary import (  # noqa: E402
    AnalysisConfig,
    analyze_morphologies,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract baseline-free load response and spatial separability from "
            "format-v3 physical contact sessions."
        )
    )
    parser.add_argument(
        "sessions", nargs="+", type=Path, help="format-v3 session directories"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="analysis output directory"
    )
    parser.add_argument(
        "--expected-repetitions",
        type=int,
        default=5,
        help="expected independent runs per indenter/hole (default: 5)",
    )
    parser.add_argument(
        "--hole-spacing-mm",
        type=float,
        default=None,
        help="trusted physical spacing between neighboring hole indices",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output = analyze_morphologies(
        args.sessions,
        args.output,
        config=AnalysisConfig(
            expected_repetitions=args.expected_repetitions,
            hole_spacing_mm=args.hole_spacing_mm,
        ),
    )
    summary = output / "raw_data_summary"
    size_bytes = sum(path.stat().st_size for path in summary.rglob("*") if path.is_file())
    print(f"Results: {output / 'results'}")
    print(f"Figures: {output / 'figures'}")
    print(f"Raw data summary: {summary} ({size_bytes / 1024**2:.3f} MiB)")
    print(f"Upload archive: {output / 'raw_data_summary.zip'}")


if __name__ == "__main__":
    main()
