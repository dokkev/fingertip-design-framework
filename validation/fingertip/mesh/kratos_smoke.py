"""Initialize the Phase 4M fingertip mesh and three internal ALM contact pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mesh.fingertip import generate_fingertip_mesh
from fem.kratos_adapter import run_initialization_smoke
from mesh.types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh-level", choices=("medium", "fine"), default="medium"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/validation/fingertip/mesh/kratos_smoke_medium.json"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    model = FingertipModel(FingertipParameters())
    mesh = generate_fingertip_mesh(
        model, mesh_settings_for_level(arguments.mesh_level)
    )
    result = run_initialization_smoke(mesh)
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Phase 4M Kratos initialization: {result['status']}")
    print(output)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
