# Scientific visualization layer

`visualization/` is a thin Matplotlib consumer of neutral model, mesh, FEA,
and optical result data. It never starts Gmsh, Kratos, OptiX, or Mitsuba and it
does not calculate evaluation metrics or alter raw arrays.

The public plot facade is:

```python
from visualization import (
    plot_camera,
    plot_case_comparison,
    plot_fea,
    plot_fingertip,
    plot_mesh,
    plot_transport,
)
```

The implementation has three deliberately small levels:

- `draw_*` functions add one physical or scalar layer to an existing Axes.
  They do not choose limits, aspect, labels, colorbars, or figure layout.
- `plot_*` functions create an optional standalone Axes, call the draw layers,
  apply physical-axis policy, and own standalone colorbars/titles.
- `plot_case_comparison` owns only the 2×2 layout, shared physical bounds,
  shared norms, row colorbars, titles, and calls to the same geometry,
  mechanics, and optics draw layers.

There is no generic figure DSL, panel data model, scene graph, or export
framework. A scientific workflow under `validation/figures/` remains an
explicit consumer that owns artifact selection, provenance, labels, and
`savefig` calls.

## Optical paper view

The optical paper panels consume the result-owned PLANAR_2D projected field and
`projected_optical_mask`. They build one `PowerNorm(gamma=0.45)` from the
positive in-domain values of both unloaded and loaded fields, using the shared
99.5th percentile as `vmax`. The fields are never independently normalized.

The raster shown in a figure is a display copy: cells outside the physical
optical domain and extremely small path-density values are masked, and a
one-cell domain-aware raster filter suppresses bin/ray aliasing. This smoothing
and low-weight floor are visualization-only; optimization and evaluation use
the raw result field.

The default paper optical layers are limited to the scalar field, clean
fingertip outline, LED/source marker, and loaded contact boundary. Exit points,
quivers, and representative retained paths are available through the explicit
debug view (`show_exits=True` in the case composer or `debug=True`/`show_rays`
in `plot_transport`). Debug paths are deterministically bounded to about 100
primary paths.

Production PLANAR_2D results expose the field, grid edges, and mask directly;
when segment retention is enabled they also expose neutral retained segment
starts, ends, media, weights, and primary-ray indices for debug rendering.

## Existing figure workflow

The normal-indentation displacement atlas is rendered with:

```bash
python -m validation.figures.displacement_atlas \
  --input-dir output/validation/fingertip/indentation/normal_full_field \
  --output output/figures/displacement_vector_atlas/displacement_vector_atlas.png
```

Generated figures belong under `output/` and remain untracked.
