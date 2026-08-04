"""Run/reuse full-field FEM cases, then render their persisted dataset."""

from __future__ import annotations

if __package__:
    from .bootstrap import ensure_repository_root
else:
    from bootstrap import ensure_repository_root

repository_root = ensure_repository_root()

from validation.fingertip.indentation.normal_field_atlas import (
    run_normal_field_atlas,
)
from visualization.framework import (
    load_figure_spec,
    load_visualization_dataset,
    show_figure,
)


def main() -> int:
    manifest = run_normal_field_atlas()
    spec = load_figure_spec(
        repository_root / "examples" / "displacement_vector_atlas.yaml"
    )
    dataset = load_visualization_dataset(spec)
    print(manifest)
    show_figure(dataset, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
