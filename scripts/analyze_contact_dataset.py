#!/usr/bin/env python3
"""Analyze one or more fixed-camera LUMO contact-dataset sessions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.pipeline import AnalysisConfig, analyze_sessions  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and compare compact physical-contact dataset features."
    )
    parser.add_argument(
        "sessions", nargs="+", type=Path, help="format-v3 session directories"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="analysis output directory"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="discard reusable extracted-feature caches and decode raw PNGs again",
    )
    parser.add_argument(
        "--expected-repetitions",
        type=int,
        default=5,
        help="expected independent repetitions per indenter/hole (default: 5)",
    )
    parser.add_argument(
        "--hole-spacing-mm",
        type=float,
        default=None,
        help="trusted physical spacing between adjacent hole indices, if known",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = AnalysisConfig(
        expected_repetitions=args.expected_repetitions,
        hole_spacing_mm=args.hole_spacing_mm,
    )
    bundle = analyze_sessions(
        args.sessions,
        args.output,
        recompute=args.recompute,
        config=config,
    )
    print(f"Analysis bundle: {bundle}")
    print(f"Archive: {bundle.parent / 'analysis_bundle.zip'}")


if __name__ == "__main__":
    main()
