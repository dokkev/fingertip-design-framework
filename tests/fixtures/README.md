# Test fixtures

Keep only small, deterministic inputs needed by `tests/unit` or `tests/smoke`
here. Generated validation artifacts belong under `output/`, and scientific
reference datasets belong under `validation/reference_data/`.

Fingertip geometry fixtures must use the component-specific
`vertical_pad_*` and `semielliptical_pad_*` schema and must keep the complete
stem-clearance cutout inside the outer pad envelope. The canonical nonzero
clearance fixture uses `void_width=1.0` and `void_height=2.0`. Artifacts
containing the obsolete `pad_width` or `pad_height` schema describe the old
single semi-ellipse and must be regenerated rather than reused.
