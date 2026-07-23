"""Command-line entry point for declarative LIT Hand scientific figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from visualization.data import ScientificFigureError
from visualization.framework import (
    load_figure_spec,
    load_visualization_dataset,
    render_figure,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a geometry-aware LIT Hand figure from a JSON or "
            "JSON-compatible YAML FigureSpec."
        )
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the spec output directory; matching files are overwritten.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        spec = load_figure_spec(args.spec)
        dataset = load_visualization_dataset(spec)
        result = render_figure(
            dataset,
            spec,
            output_directory=args.output_dir,
        )
    except ScientificFigureError as exc:
        print(f"scientific figure FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
