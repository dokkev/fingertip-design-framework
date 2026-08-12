# Test fixtures

Keep only small, deterministic inputs needed by `tests/unit` or `tests/smoke`
here. Generated validation artifacts belong under `output/`, and scientific
reference datasets belong under `validation/reference_data/`.

Fingertip geometry fixtures must use the canonical `flat_pad_*`, explicit
`link_width`, and `bond_extension_height` schema. The complete stem-clearance
cutout must remain inside the outer pad envelope. The canonical nonzero
clearance fixture uses `void_width=1.0` and `void_height=2.0`. Artifacts
containing obsolete single-pad schemas must be regenerated rather than reused.
