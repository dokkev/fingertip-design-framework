#!/usr/bin/env python3
"""Analyze cyclic contact-history sessions for one material."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.contact_history import analyze_contact_history  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare actual-force-matched optical history and same-contact "
            "cycle repeatability across three morphologies of one material."
        )
    )
    parser.add_argument(
        "sessions",
        nargs=3,
        type=Path,
        help="baseline, flat-opt, and angled-opt contact-history sessions",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat-metrics",
        type=Path,
        default=None,
        help="optional morphology_metrics.csv containing 10 mm W_repeat",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output = analyze_contact_history(
        args.sessions,
        args.output,
        repeat_metrics_path=args.repeat_metrics,
    )
    print(f"Report: {output / 'report.md'}")
    print(f"Results: {output / 'results'}")
    print(f"Figures: {output / 'figures'}")


if __name__ == "__main__":
    main()
