# Visualization ownership

`visualization/` is intentionally a small Matplotlib layer. Its public API is:

```python
from visualization import (
    plot_camera,
    plot_displacement,
    plot_fingertip,
    plot_mesh,
    plot_transport,
)
```

Each helper consumes an object already owned by another package, draws on an
optional Matplotlib `Axes`, and returns that `Axes`. The package does not load
validation artifacts, reconstruct topology, start Gmsh/Kratos, import
Mitsuba at package import time, calculate scientific metrics, or serialize a
figure specification.

`plot_fingertip` accepts the public `model.Fingertip` facade. `plot_mesh` and
`plot_displacement` consume the neutral `mesh.PadMesh` contract (and use the
private `.pad` view when passed a full `FingertipMesh`). Displacement magnitude
is shown in millimeters on the deformed T3 connectivity; arrows remain the
physical displacement vectors and are deterministically limited only for
display. `plot_transport` visualizes the raw `TransportResult.density` on a
normalized display scale without changing the result. `plot_camera` copies
and normalizes `linear_rgb` for display, so camera results remain raw.

Scientific figure workflows belong in `validation/figures/` or the relevant
validation package. They own artifact manifests, checksums, case selection,
scientific labels, direct `savefig` calls, and any publication-specific
layout. The normal-indentation displacement atlas is rendered with:

```bash
python -m validation.figures.displacement_atlas \
  --input-dir output/validation/fingertip/indentation/normal_full_field \
  --output output/figures/displacement_vector_atlas/displacement_vector_atlas.png
```

The Phase 4K transfer-map workflow lives in
`validation/figures/transfer_map.py` and writes its plots from canonical
arrays. Reproducibility is provided by tracked Python code, persisted
artifacts, and explicit command arguments rather than a generic figure DSL.
